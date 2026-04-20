# mujoco-cli-vision

A vision module for [mujoco-cli](https://github.com/yamatsushita/mujoco-cli) that gives the Copilot CLI **eyes**.

Instead of hard-coding object names and positions, the system captures a scene image from a fixed camera and uses **Florence-2** (Microsoft, MIT licence) to detect objects, generate descriptions, and optionally localise them in 3-D world space.  The resulting scene context is automatically injected into every Copilot CLI prompt so the model understands the scene without prior knowledge.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  GitHub Issue (browser)                                          │
│    User: "move the cube to the right of the sphere"              │
└───────────────────────────┬──────────────────────────────────────┘
                            │ poll (client_vision.py)
┌───────────────────────────▼──────────────────────────────────────┐
│  VisionCLIClient                                                 │
│    1. GET /scene  →  vision server                               │
│         └─ returns: "1. red cube centre (0.42, 0.61) …"         │
│    2. Augmented prompt:                                          │
│         [scene context block]                                    │
│         ---                                                      │
│         User request: move the cube to the right of the sphere   │
│    3. gh copilot -p "<augmented prompt>"  →  MuJoCo code        │
└───────────────────────────┬──────────────────────────────────────┘
                            │ execute
┌───────────────────────────▼──────────────────────────────────────┐
│  MuJoCo simulation                                               │
│    Fixed camera renders scene  →  vision server analyses it      │
└──────────────────────────────────────────────────────────────────┘
```

## Components

| File | Role |
|---|---|
| `vision/analyzer.py` | Florence-2 wrapper — object detection, captioning, phrase grounding |
| `vision/capture.py` | MuJoCo renderer + camera-based 2-D→3-D unprojection |
| `vision/server.py` | FastAPI REST server exposing the vision pipeline |
| `client_vision.py` | Extended `mujoco-cli.py` with automatic scene injection |
| `examples/analyze_image.py` | Standalone image analysis demo |
| `examples/mujoco_integration.py` | Full end-to-end MuJoCo demo |

---

## Why Florence-2?

[Florence-2](https://huggingface.co/microsoft/Florence-2-large) (Microsoft, 2024, MIT licence) is a unified vision-language foundation model that handles object detection, dense region captioning, phrase grounding, and scene description from a single prompt-based interface.

| Property | Value |
|---|---|
| Parameters | 0.23 B (base) / 0.77 B (large) |
| Licence | MIT |
| HuggingFace | `microsoft/Florence-2-large` |
| Tasks used | `<OD>`, `<DETAILED_CAPTION>`, `<DENSE_REGION_CAPTION>`, `<OPEN_VOCABULARY_DETECTION>` |
| GPU required | No — runs on CPU (slower) or GPU (recommended) |
| Internet required | First run only (downloads model weights, ~1.5 GB) |

---

## Quick start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
# For live MuJoCo capture:
pip install mujoco
```

### 2 · Start the vision server

```bash
# Analysis only (no live MuJoCo capture):
python -m vision.server --port 8765

# With live MuJoCo capture:
python -m vision.server --port 8765 --xml /path/to/your/model.xml

# Use the lighter model to save memory:
python -m vision.server --model microsoft/Florence-2-base
```

The first startup downloads Florence-2 weights from HuggingFace (~1.5 GB).  Subsequent starts are instant.

### 3 · Start the vision client

```bash
# Link client_vision.py to the mujoco-cli directory so it can find mujoco-cli.py,
# or set PYTHONPATH:
export PYTHONPATH=/path/to/mujoco-cli

python client_vision.py \
    --token ghp_xxx \
    --name desktop \
    --vision-url http://localhost:8765
```

The client automatically injects the current scene context into every Copilot prompt.  No changes to your prompt style are needed.

---

## REST API

### `GET /health`
```json
{"status":"ok","model_loaded":true,"capture_available":true,"scene_cached":false}
```

### `POST /analyze`
Upload an image file (multipart `file` field) or pass `image_b64` query param.

```bash
curl -X POST http://localhost:8765/analyze \
     -F "file=@scene.png"
```

Response:
```json
{
  "scene": {
    "caption": "A tabletop with a red cube and a blue sphere.",
    "objects": [
      {"label": "red cube", "bbox_pixels": [120, 200, 220, 300],
       "center_norm": [0.27, 0.52], "area_norm": 0.032},
      ...
    ],
    "image_size": [640, 480]
  },
  "context": "## Scene perception …\n …"
}
```

### `GET /scene`
Returns the cached scene from the last `/analyze` or `/capture` call.

### `POST /query?text=red+cube.+blue+sphere`
Text-conditioned detection on the cached image.

### `POST /capture?camera=fixed_cam&with_depth=true`
Render the current MuJoCo scene and analyse it (requires `--xml` on server start).

---

## Built-in client commands

| Command | Description |
|---|---|
| `\scene` | Show the current cached scene description |
| `\capture [cam]` | Render the MuJoCo scene and analyse it |
| `\analyze <path>` | Analyse a local image file |
| `\query <text>` | Find specific objects in the scene |
| `\vision on/off` | Enable/disable automatic scene injection |

All standard `mujoco-cli.py` commands (`\ping`, `\shell`, `\clear`, etc.) are available unchanged.

---

## 3-D localisation

When the vision server is started with `--xml` and you call `/capture?with_depth=true`, the server:

1. Renders an RGB image **and** a depth map from MuJoCo.
2. Detects objects with Florence-2.
3. Reads the depth at each bounding-box centre.
4. Unprojects the pixel + depth to world coordinates using MuJoCo's camera intrinsics/extrinsics.

The resulting `world_xyz` values are appended to each object in the response:

```json
{"label": "red cube", "center_norm": [0.27, 0.52],
 "world_xyz": [0.312, -0.045, 0.101]}
```

These 3-D coordinates are included in the scene context injected into Copilot prompts, giving the model accurate spatial information to generate correct robot control code.

---

## Integrating with an existing MuJoCo controller

1. In your simulation loop, call `capture.capture()` after each environment step (or on demand).
2. POST the image to `/analyze`.
3. The vision server caches the analysis; `client_vision.py` polls `/scene` before each Copilot call.

```python
# Example: post a frame from your simulation loop
import requests
from PIL import Image
import io

def update_vision_server(image: Image.Image, vision_url="http://localhost:8765"):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    requests.post(f"{vision_url}/analyze", files={"file": buf}, timeout=60)
```

---

## Licence

MIT
