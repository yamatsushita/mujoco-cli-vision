# mujoco-cli-vision

Vision-augmented extension of [mujoco-cli](https://github.com/yamatsushita/mujoco-cli).

Instead of relying on hard-coded object positions, a local **Florence-2**
vision model renders the MuJoCo scene before every LLM planning call and
describes what it sees.  The Copilot CLI planner receives only visual
information (detected objects, bounding boxes, scene caption) — it has no
prior knowledge of object names or locations.

---

## How it works

```
User instruction
      │
      ▼
mujoco-cli-vision.py          (in-process, no server)
  ┌─────────────────────────────────────────────────────────┐
  │  1. env.render() → numpy RGB frame                      │
  │  2. Florence-2 analyzes the frame:                      │
  │       <DETAILED_CAPTION>  → scene overview              │
  │       <OD>                → detected objects + bboxes   │
  │  3. Visual scene description replaces the programmatic  │
  │     describe_scene() call in mujoco-cli's agent module  │
  │  4. Enriched prompt → copilot -p "…"                   │
  │  5. Plan executed in MuJoCo                             │
  └─────────────────────────────────────────────────────────┘
```

Florence-2 is a unified vision-language model (0.77 B parameters, MIT
licence) from Microsoft that runs **fully locally** — no API key or cloud
service required.

---

## Requirements

```
pip install -r requirements.txt
```

You also need:
- [mujoco-cli](https://github.com/yamatsushita/mujoco-cli) checked out locally
- `copilot` CLI in PATH (`gh extension install github/gh-copilot`, then `gh copilot`)

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

---

## Optional: standalone vision server

A FastAPI server is also provided for use cases where the vision pipeline
needs to run as a separate process (e.g., on a remote machine):

```bash
python -m vision.server --port 8765
curl http://localhost:8765/health
```

This is **not** required for normal use of `mujoco-cli-vision.py`.

---

## Project layout

```
mujoco-cli-vision/
├── mujoco-cli-vision.py   Entry point — loads Florence-2, patches mujoco-cli
├── vision/
│   ├── analyzer.py        Florence-2 wrapper (SceneAnalyzer, SceneAnalysis)
│   ├── capture.py         MuJoCo renderer + 3-D unprojection helpers
│   ├── server.py          Optional FastAPI server
│   └── __init__.py
├── examples/
│   ├── analyze_image.py   Standalone image analysis demo
│   └── mujoco_integration.py  End-to-end demo
└── requirements.txt
```

---

## Florence-2 compatibility

Florence-2's remote code has known incompatibilities with transformers ≥ 5.x.
`SceneAnalyzer` automatically patches the cached model files on first load:

| Symptom | Fix applied |
|---------|-------------|
| `AttributeError: forced_bos_token_id` | Patches `configuration_florence2.py` |
| `AttributeError: additional_special_tokens` | Patches `processing_florence2.py` |
| `AttributeError: _supports_sdpa` | Passes `attn_implementation="eager"` |
