# mujoco-cli-vision

Vision-augmented extension of [mujoco-cli](https://github.com/yamatsushita/mujoco-cli).

Instead of relying on hard-coded object positions, this project uses
**RGB+D** images from the MuJoCo renderer to perceive the scene without
any pre-defined knowledge:

- **Florence-2** analyses the RGB image to detect objects and generate
  scene captions (object labels, bounding boxes).
- **FoundationPose** (or a depth-based fallback) estimates 6-DoF object
  poses from the RGB+D data (position + orientation in world coordinates).
- The combined perception is fed to the **Copilot CLI** planner which
  produces a sequence of robot actions.
- Actions are executed and **visualised in MuJoCo**.

---

## How it works

```
User instruction
      │
      ▼
mujoco-cli-vision.py          (in-process, no server)
  ┌─────────────────────────────────────────────────────────┐
  │  1. vision_cam renders RGB + Depth from MuJoCo          │
  │  2. Florence-2 analyses RGB:                            │
  │       <DETAILED_CAPTION>  → scene overview              │
  │       <OD>                → detected objects + bboxes   │
  │  3. FoundationPose / depth fallback analyses RGB+D:     │
  │       6-DoF poses → world position + orientation        │
  │  4. Combined scene description replaces describe_scene  │
  │     and _build_action_reference in mujoco-cli           │
  │  5. Enriched prompt → copilot -p "…"                   │
  │  6. Plan executed in MuJoCo + visualised in viewer      │
  └─────────────────────────────────────────────────────────┘
```

A dedicated `vision_cam` camera is injected into the MuJoCo scene XML
to observe the entire workspace (arm + table + objects) from a slightly
elevated 3/4 angle, providing good depth contrast for pose estimation.

---

## Requirements

```
pip install -r requirements.txt
```

You also need:
- [mujoco-cli](https://github.com/yamatsushita/mujoco-cli) checked out locally
- `copilot` CLI in PATH (`gh extension install github/gh-copilot`, then `gh copilot`)
- **Optional**: [FoundationPose](https://github.com/NVlabs/FoundationPose) for full 6-DoF
  pose estimation (requires NVIDIA GPU + CUDA). Without it, the system uses
  depth-based position estimation (position only, no orientation).

---

## Usage

```bash
# Single instruction
python mujoco-cli-vision.py \
    --mujoco-cli /path/to/mujoco-cli \
    "Pick up the red cube"

# Interactive mode (UNDO / CLEAR supported)
python mujoco-cli-vision.py \
    --mujoco-cli /path/to/mujoco-cli \
    --interactive

# Use the lighter base model (faster on CPU)
python mujoco-cli-vision.py \
    --mujoco-cli /path/to/mujoco-cli \
    --model microsoft/Florence-2-base \
    "Stack the cubes"

# Disable FoundationPose (use depth fallback only)
python mujoco-cli-vision.py \
    --mujoco-cli /path/to/mujoco-cli \
    --no-foundation-pose \
    "Pick up the red cube"

# Set mujoco-cli path via environment variable
export MUJOCO_CLI_PATH=/path/to/mujoco-cli
python mujoco-cli-vision.py "Pick up the red cube"
```

All `mujoco-cli.py` flags are forwarded unchanged:

| Flag | Description |
|------|-------------|
| `--scene N` | Scene preset 0–5 |
| `--seed N` | Random seed for object layout |
| `--max-retries N` | Replan up to N times on failure |
| `--no-viewer` | Disable MuJoCo viewer window |
| `--output path.mp4` | Record video |
| `--interactive` | Interactive prompt loop |

### Vision-specific flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mujoco-cli PATH` | `$MUJOCO_CLI_PATH` | Path to mujoco-cli directory |
| `--model MODEL_ID` | `microsoft/Florence-2-large` | Florence-2 HuggingFace model ID |
| `--device DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `--no-foundation-pose` | (off) | Disable FoundationPose, use depth fallback |

---

## Perception pipeline

### Input: RGB+D from MuJoCo

The `vision_cam` camera renders both an RGB image and a depth map from the
MuJoCo simulation.  The camera is positioned at `(1.30, -0.75, 1.55)` with
a 52° field of view, providing full coverage of the Franka Panda arm and
the manipulation table.

### Florence-2 (object detection)

The RGB image is passed to Florence-2, which produces:
- A detailed scene caption describing the overall layout
- Object detection with labels and bounding boxes (`<OD>` task)

### FoundationPose / depth fallback (pose estimation)

The RGB image, depth map, and Florence-2 bounding boxes are passed to the
pose estimator:
- **FoundationPose mode** (`full`): Full 6-DoF pose (position + orientation)
  using NVIDIA's FoundationPose with foreground masks extracted from
  bounding boxes + depth segmentation.
- **Depth fallback mode** (`position_only`): 3D position estimated by
  unprojecting the median depth within each bounding box through the
  camera's pinhole model.  Orientation is reported as unknown.

### Output to Copilot CLI

The planner receives a scene description containing:
```
=== Visual Scene (Florence-2 + pose estimation) ===
Perception mode: depth_fallback
Scene overview: A robotic arm next to a table with coloured blocks...

Detected objects with estimated poses:
   1. 'red cube' [pos] position=(0.550, 0.100, 0.575), orientation=unknown, conf=70%
   2. 'blue cube' [pos] position=(0.550, -0.100, 0.575), orientation=unknown, conf=70%
```

---

## Project layout

```
mujoco-cli-vision/
├── mujoco-cli-vision.py       Entry point — RGB+D pipeline, patches mujoco-cli
├── vision/
│   ├── analyzer.py            Florence-2 wrapper (SceneAnalyzer, SceneAnalysis)
│   ├── capture.py             MuJoCo renderer + depth + 3-D unprojection
│   ├── pose_estimator.py      FoundationPose wrapper + depth fallback
│   ├── server.py              Optional FastAPI server
│   └── __init__.py
├── examples/
│   ├── analyze_image.py       Standalone image analysis demo
│   └── mujoco_integration.py  End-to-end demo
└── requirements.txt
```

---

## FoundationPose setup (optional)

For full 6-DoF pose estimation, install FoundationPose:

```bash
git clone https://github.com/NVlabs/FoundationPose.git
cd FoundationPose
# Follow their installation instructions (requires Docker + NVIDIA GPU)
pip install -e .
```

Without FoundationPose, the system automatically falls back to depth-based
position estimation, which is sufficient for most pick-and-place tasks.

---

## Florence-2 compatibility

Florence-2's remote code has known incompatibilities with transformers ≥ 5.x.
`SceneAnalyzer` automatically patches the cached model files on first load:

| Symptom | Fix applied |
|---------|-------------|
| `AttributeError: forced_bos_token_id` | Patches `configuration_florence2.py` |
| `AttributeError: additional_special_tokens` | Patches `processing_florence2.py` |
| `AttributeError: _supports_sdpa` | Passes `attn_implementation="eager"` |
