"""
vision/server.py
================
FastAPI REST server that exposes the vision pipeline to the remote CLI client
and to any other consumer (e.g. a MuJoCo controller script).

Endpoints
---------
GET  /health
    Liveness check.  Returns model-loaded flag.

POST /analyze
    Analyse an uploaded image.
    Body: multipart ``file`` (image) **or** JSON ``{"image_b64": "<base64>"}``
    Returns: SceneAnalysis dict + formatted context string.

GET  /scene
    Return the last cached scene analysis (populated by /analyze or /capture).

POST /query
    Text-conditioned detection on the last cached image.
    Query param: ``text`` (e.g. "red cube. blue sphere")
    Optionally upload a new ``file`` to override the cached image.

POST /capture
    Trigger a live MuJoCo render and analyse it.
    Query params: width, height, camera (int or name string)
    Requires the server to have been started with ``--xml`` flag.

Usage
-----
    # Start the server (installs dependencies first time):
    uvicorn vision.server:app --host 0.0.0.0 --port 8765

    # Or use the CLI helper:
    python -m vision.server --port 8765 --model microsoft/Florence-2-base

Environment variables
---------------------
    FLORENCE_MODEL   Override HuggingFace model ID (default: Florence-2-large)
    MUJOCO_XML       Path to MuJoCo XML for /capture endpoint
    VISION_PORT      Default listen port (default: 8765)
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from .analyzer import SceneAnalyzer, SceneAnalysis

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MuJoCo Vision Server",
    version="1.0.0",
    description=(
        "Florence-2 powered scene perception for MuJoCo-CLI. "
        "Analyses camera images and returns structured object descriptions."
    ),
)

# ── Global state ──────────────────────────────────────────────────────────────

_analyzer: Optional[SceneAnalyzer] = None
_last_analysis: Optional[SceneAnalysis] = None
_last_image: Optional[Image.Image] = None
_capture: Optional[object] = None   # MuJoCoCapture, loaded lazily


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    global _analyzer, _capture
    model_name = os.environ.get("FLORENCE_MODEL", SceneAnalyzer.DEFAULT_MODEL)
    logger.info("Initialising Florence-2 (%s)…", model_name)
    _analyzer = SceneAnalyzer(model_name=model_name)

    xml_path = os.environ.get("MUJOCO_XML")
    if xml_path and Path(xml_path).exists():
        try:
            from .capture import MuJoCoCapture
            _capture = MuJoCoCapture(xml_path=xml_path)
            logger.info("MuJoCo model loaded from %s", xml_path)
        except ImportError:
            logger.warning("mujoco package not available; /capture disabled.")


# ── Helper ────────────────────────────────────────────────────────────────────

def _image_from_upload(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def _analysis_response(analysis: SceneAnalysis) -> dict:
    return {
        "scene": analysis.to_dict(),
        "context": analysis.to_context_string(),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _analyzer is not None,
        "capture_available": _capture is not None,
        "scene_cached": _last_analysis is not None,
    }


@app.post("/analyze")
async def analyze(
    file: Optional[UploadFile] = File(None),
    image_b64: Optional[str] = Query(None, description="Base-64 encoded image"),
    dense: bool = Query(False, description="Use dense region captioning (slower)"),
):
    """
    Analyse an image and cache the result.

    Supply either a multipart ``file`` upload or a ``image_b64`` query
    parameter containing the base-64 encoded image bytes.
    """
    global _last_analysis, _last_image

    if _analyzer is None:
        raise HTTPException(503, "Vision model not loaded yet.")

    if file is not None:
        raw = await file.read()
        image = _image_from_upload(raw)
    elif image_b64 is not None:
        raw = base64.b64decode(image_b64)
        image = _image_from_upload(raw)
    else:
        raise HTTPException(400, "Provide 'file' (multipart) or 'image_b64' query param.")

    _last_image = image
    _last_analysis = _analyzer.analyze_scene(image, use_dense_captions=dense)
    return _analysis_response(_last_analysis)


@app.get("/scene")
def get_scene():
    """Return the cached scene from the last /analyze or /capture call."""
    if _last_analysis is None:
        raise HTTPException(404, "No scene cached yet. Call /analyze or /capture first.")
    return _analysis_response(_last_analysis)


@app.post("/query")
async def query_objects(
    text: str = Query(..., description='Object description, e.g. "red cube. blue sphere"'),
    file: Optional[UploadFile] = File(None),
):
    """
    Text-conditioned detection: find specific objects in the (cached) scene.

    Returns bounding boxes for every object matching the text description.
    """
    global _last_image

    if _analyzer is None:
        raise HTTPException(503, "Vision model not loaded yet.")

    image = _last_image
    if file is not None:
        raw = await file.read()
        image = _image_from_upload(raw)
        _last_image = image

    if image is None:
        raise HTTPException(
            404,
            "No image available. Call /analyze (or /capture) first, or upload a file.",
        )

    detections = _analyzer.ground_objects(image, text)
    return {
        "query": text,
        "matches": [d.to_dict() for d in detections],
    }


@app.post("/capture")
def capture_scene(
    width: int = Query(640),
    height: int = Query(480),
    camera: str = Query("0", description="Camera index (int) or name string"),
    dense: bool = Query(False),
    with_depth: bool = Query(False, description="Return depth-based 3-D positions"),
):
    """
    Render the current MuJoCo scene and analyse it.

    Requires the server to have been started with the ``MUJOCO_XML`` env var
    (or ``--xml`` argument to ``python -m vision.server``).
    """
    global _last_analysis, _last_image

    if _capture is None:
        raise HTTPException(
            501,
            "MuJoCo capture not available. "
            "Set MUJOCO_XML env var and restart the server.",
        )
    if _analyzer is None:
        raise HTTPException(503, "Vision model not loaded yet.")

    # Resolve camera argument: try integer first
    try:
        cam = int(camera)
    except ValueError:
        cam = camera   # named camera string

    if with_depth:
        image, depth_map = _capture.capture_with_depth(
            width=width, height=height, camera=cam
        )
    else:
        image = _capture.capture(width=width, height=height, camera=cam)
        depth_map = None

    _last_image = image
    analysis = _analyzer.analyze_scene(image, use_dense_captions=dense)
    _last_analysis = analysis

    response = _analysis_response(analysis)

    if with_depth and depth_map is not None:
        # Augment objects with 3-D world coordinates
        objects_3d = _capture.localize_objects_3d(
            [o.to_dict() for o in analysis.objects],
            depth_map,
            camera=cam,
        )
        response["objects_3d"] = objects_3d

    return response


# ── CLI entry point ───────────────────────────────────────────────────────────

def _main():
    import argparse

    parser = argparse.ArgumentParser(description="MuJoCo Vision Server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VISION_PORT", 8765)),
        help="Listen port (default: 8765)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("FLORENCE_MODEL", SceneAnalyzer.DEFAULT_MODEL),
        help="Florence-2 HuggingFace model ID",
    )
    parser.add_argument("--xml", help="MuJoCo XML model path for /capture endpoint")
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    args = parser.parse_args()

    os.environ["FLORENCE_MODEL"] = args.model
    if args.xml:
        os.environ["MUJOCO_XML"] = args.xml

    logging.basicConfig(level=args.log_level.upper())
    uvicorn.run("vision.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    _main()
