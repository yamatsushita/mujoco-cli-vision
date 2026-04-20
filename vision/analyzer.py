"""
vision/analyzer.py
==================
Scene analyzer powered by **Florence-2** (Microsoft, MIT licence).

Florence-2 is a unified vision-language foundation model (0.23 B / 0.77 B
parameters) that handles object detection, dense region captioning, phrase
grounding and more from a single, prompt-based interface.  It runs fully
locally — no API key required.

HuggingFace model cards:
  microsoft/Florence-2-base   (lighter, faster)
  microsoft/Florence-2-large  (recommended for accuracy)

Key tasks used here:
  <OD>                      – open-vocabulary object detection (boxes + labels)
  <DENSE_REGION_CAPTION>    – per-region natural-language captions
  <DETAILED_CAPTION>        – whole-image scene description
  <OPEN_VOCABULARY_DETECTION> – text-conditioned detection of named objects
"""

from __future__ import annotations

import glob as _glob
import io
import logging
import os
import sys
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

logger = logging.getLogger(__name__)


# ── Florence-2 config cache patcher ──────────────────────────────────────────

def _patch_florence2_config_cache() -> bool:
    """
    Patch the cached configuration_florence2.py to fix the
    ``'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'``
    error that occurs with transformers >= 4.49 (changed PretrainedConfig
    __getattribute__ no longer silently returns None for unset attributes).

    The bug is inside the *model's own remote code* (not in transformers itself):
    ``Florence2LanguageConfig.__init__`` accesses ``self.forced_bos_token_id``
    before calling ``super().__init__()``, so the attribute has not been set yet.

    This function finds all copies of the file in the HuggingFace modules cache,
    replaces the bare ``self.forced_bos_token_id`` reference with a safe
    ``getattr(self, 'forced_bos_token_id', None)``, removes the stale .pyc
    files, and evicts any partially loaded entries from sys.modules.

    Returns True if at least one file was patched.
    """
    BAD  = "if self.forced_bos_token_id is None"
    GOOD = "if getattr(self, 'forced_bos_token_id', None) is None"

    # Resolve HuggingFace cache root (respects HF_HOME / HF_HUB_CACHE env vars)
    hf_cache = Path(
        os.environ.get("HF_HOME", "")
        or os.environ.get("HF_HUB_CACHE", "")
        or os.environ.get("HUGGINGFACE_HUB_CACHE", "")
        or (Path.home() / ".cache" / "huggingface")
    )
    modules_dir = hf_cache / "modules" / "transformers_modules"

    patched_any = False
    for path_str in _glob.glob(
        str(modules_dir / "**" / "configuration_florence2.py"), recursive=True
    ):
        cfg = Path(path_str)
        try:
            text = cfg.read_text(encoding="utf-8")
        except OSError:
            continue

        if BAD not in text or GOOD in text:
            continue  # Not buggy or already patched

        cfg.write_text(text.replace(BAD, GOOD), encoding="utf-8")
        logger.info("Patched Florence-2 cached config: %s", cfg)
        patched_any = True

        # Remove compiled bytecode so Python recompiles from the patched source
        pycache = cfg.parent / "__pycache__"
        if pycache.exists():
            for pyc in pycache.glob("configuration_florence2*.pyc"):
                try:
                    pyc.unlink()
                except OSError:
                    pass

    # Evict any partially loaded module entries from Python's import cache
    for key in list(sys.modules):
        if "configuration_florence2" in key or "florence2" in key.lower():
            del sys.modules[key]

    return patched_any


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class DetectedObject:
    """A single object detected in the scene."""
    label: str
    bbox_pixels: list[float]          # [x1, y1, x2, y2] in image pixels
    center_norm: list[float]          # [cx, cy] normalised to [0, 1]
    area_norm: float                  # fraction of image area
    caption: Optional[str] = None     # rich description if available

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "bbox_pixels": self.bbox_pixels,
            "center_norm": self.center_norm,
            "area_norm": self.area_norm,
            "caption": self.caption,
        }


@dataclass
class SceneAnalysis:
    """Result of a full scene analysis."""
    caption: str
    objects: list[DetectedObject] = field(default_factory=list)
    image_size: list[int] = field(default_factory=lambda: [0, 0])   # [W, H]

    def to_dict(self) -> dict:
        return {
            "caption": self.caption,
            "objects": [o.to_dict() for o in self.objects],
            "image_size": self.image_size,
        }

    def to_context_string(self) -> str:
        """
        Produce a compact, human-readable block that can be prepended to a
        Copilot CLI prompt so the model understands the scene without having
        been given prior knowledge.
        """
        lines = [
            "## Scene perception (auto-generated by vision module)",
            "",
            f"**Scene overview:** {self.caption}",
            "",
            "**Detected objects** (image coordinates: origin = top-left,"
            " values normalised to 0–1):",
        ]
        if self.objects:
            for i, obj in enumerate(self.objects, start=1):
                cx, cy = obj.center_norm
                area_pct = obj.area_norm * 100
                desc = obj.caption or obj.label
                lines.append(
                    f"  {i:2d}. `{obj.label}` — centre ({cx:.2f}, {cy:.2f}),"
                    f" area {area_pct:.1f}%"
                    + (f"\n       ↳ {obj.caption}" if obj.caption else "")
                )
        else:
            lines.append("  _(no objects detected)_")
        lines += ["", "---", ""]
        return "\n".join(lines)


# ── Main analyzer ─────────────────────────────────────────────────────────────

class SceneAnalyzer:
    """
    Wraps Florence-2 for MuJoCo scene understanding.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Use ``"microsoft/Florence-2-base"`` for lower
        memory footprint or ``"microsoft/Florence-2-large"`` for best accuracy.
    device:
        ``"auto"`` selects CUDA when available, otherwise CPU.
        Pass ``"cpu"``, ``"cuda"``, or ``"mps"`` to override.
    dtype:
        Torch dtype.  ``None`` → float16 on GPU, float32 on CPU.
    """

    DEFAULT_MODEL = "microsoft/Florence-2-large"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        dtype: Optional[torch.dtype] = None,
    ):
        self.device = self._resolve_device(device)
        self.dtype = dtype or (
            torch.float16 if self.device != "cpu" else torch.float32
        )
        logger.info("Loading Florence-2 from %s on %s …", model_name, self.device)
        self.model = self._load_model(model_name)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Florence-2 ready.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(spec: str) -> str:
        if spec == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return spec

    def _load_model(self, model_name: str):
        """
        Load the Florence-2 model, auto-patching the HuggingFace modules cache
        if the ``forced_bos_token_id`` AttributeError is encountered.

        On first call the model is downloaded and the custom
        ``configuration_florence2.py`` is executed.  Newer versions of
        ``transformers`` changed ``PretrainedConfig.__getattribute__`` so that
        accessing an attribute before ``super().__init__()`` raises
        ``AttributeError`` instead of silently returning ``None``.
        The Florence-2 remote code has a pre-``super().__init__()`` access of
        ``self.forced_bos_token_id``, which triggers the error.

        Strategy:
          1. Try ``from_pretrained`` normally.
          2. If the specific ``AttributeError`` is raised, the cache file now
             exists (the download succeeded before execution failed).
          3. Patch the cached ``configuration_florence2.py`` and clear Python's
             module cache.
          4. Retry — the patched file is loaded this time.
        """
        def _do_load():
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            ).to(self.device)

        try:
            return _do_load()
        except AttributeError as exc:
            if "forced_bos_token_id" not in str(exc):
                raise
            logger.warning(
                "Florence-2 config incompatibility detected "
                "(forced_bos_token_id AttributeError). "
                "Patching cached configuration_florence2.py and retrying…"
            )
            if not _patch_florence2_config_cache():
                raise RuntimeError(
                    "Could not locate the cached configuration_florence2.py "
                    "to apply the forced_bos_token_id patch.\n"
                    "Try clearing the HuggingFace cache and restarting:\n"
                    "  Windows: rmdir /s /q %USERPROFILE%\\.cache\\huggingface\\modules\n"
                    "  Linux/macOS: rm -rf ~/.cache/huggingface/modules"
                ) from exc
            return _do_load()

    def _run_task(
        self,
        image: Image.Image,
        task_token: str,
        text_input: str = "",
        max_new_tokens: int = 1024,
    ) -> dict:
        """Run a single Florence-2 task and return the post-processed dict."""
        prompt = task_token if not text_input else f"{task_token} {text_input}"
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        # Move all tensor inputs to device/dtype
        pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=3,
                do_sample=False,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        return self.processor.post_process_generation(
            generated_text,
            task=task_token,
            image_size=(image.width, image.height),
        )

    @staticmethod
    def _bbox_to_object(
        bbox: list[float],
        label: str,
        img_w: int,
        img_h: int,
        caption: Optional[str] = None,
    ) -> DetectedObject:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        area = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
        return DetectedObject(
            label=label,
            bbox_pixels=bbox,
            center_norm=[round(cx, 4), round(cy, 4)],
            area_norm=round(area, 4),
            caption=caption,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def caption(self, image: Image.Image) -> str:
        """Return a single-sentence detailed description of the whole image."""
        result = self._run_task(image, "<DETAILED_CAPTION>")
        return result.get("<DETAILED_CAPTION>", "").strip()

    def detect_objects(self, image: Image.Image) -> list[DetectedObject]:
        """
        Open-vocabulary object detection.

        Uses the ``<OD>`` task which returns bounding boxes and class labels
        for all prominent objects Florence-2 can identify.
        """
        result = self._run_task(image, "<OD>")
        od = result.get("<OD>", {})
        bboxes = od.get("bboxes", [])
        labels = od.get("labels", [])
        return [
            self._bbox_to_object(bbox, lbl, image.width, image.height)
            for bbox, lbl in zip(bboxes, labels)
        ]

    def dense_captions(self, image: Image.Image) -> list[DetectedObject]:
        """
        Dense region captioning: returns per-region natural language captions
        together with their bounding boxes.  Richer than ``detect_objects``
        but slower.
        """
        result = self._run_task(image, "<DENSE_REGION_CAPTION>")
        drc = result.get("<DENSE_REGION_CAPTION>", {})
        bboxes = drc.get("bboxes", [])
        labels = drc.get("labels", [])
        return [
            self._bbox_to_object(bbox, lbl, image.width, image.height, caption=lbl)
            for bbox, lbl in zip(bboxes, labels)
        ]

    def ground_objects(
        self, image: Image.Image, text_query: str
    ) -> list[DetectedObject]:
        """
        Text-conditioned open-vocabulary detection.

        Finds instances of objects described by ``text_query``
        (e.g. ``"red cube. blue sphere. robot gripper"``).

        Use ``.`` or ``|`` as separators for multiple object types.
        """
        result = self._run_task(
            image, "<OPEN_VOCABULARY_DETECTION>", text_query
        )
        ovd = result.get("<OPEN_VOCABULARY_DETECTION>", {})
        bboxes = ovd.get("bboxes", [])
        labels = ovd.get("labels", [])
        return [
            self._bbox_to_object(bbox, lbl, image.width, image.height)
            for bbox, lbl in zip(bboxes, labels)
        ]

    def analyze_scene(
        self,
        image: Image.Image,
        use_dense_captions: bool = False,
    ) -> SceneAnalysis:
        """
        Full scene analysis pipeline.

        1. Whole-image caption  (``<DETAILED_CAPTION>``)
        2. Object detection     (``<OD>`` or ``<DENSE_REGION_CAPTION>``)

        Parameters
        ----------
        use_dense_captions:
            When True, uses ``<DENSE_REGION_CAPTION>`` which is slower but
            provides richer per-object descriptions.
        """
        caption = self.caption(image)
        objects = (
            self.dense_captions(image)
            if use_dense_captions
            else self.detect_objects(image)
        )
        return SceneAnalysis(
            caption=caption,
            objects=objects,
            image_size=[image.width, image.height],
        )

    # ── Convenience: load image from various sources ──────────────────────────

    @staticmethod
    def load_image(source) -> Image.Image:
        """
        Accept a file path, bytes, BytesIO, or PIL Image and return RGB PIL Image.
        """
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, (str, Path)):
            return Image.open(source).convert("RGB")
        if isinstance(source, (bytes, bytearray)):
            return Image.open(io.BytesIO(source)).convert("RGB")
        if isinstance(source, io.IOBase):
            return Image.open(source).convert("RGB")
        raise TypeError(f"Unsupported image source type: {type(source)}")
