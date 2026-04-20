#!/usr/bin/env python3
"""
mujoco-cli-vision.py
====================
Vision-augmented drop-in replacement for mujoco-cli.py.

Instead of reading object positions from the simulation directly, a
Florence-2 vision model renders and analyses the scene image before
every LLM planning call.  The planner receives only visual information
(detected objects, bounding boxes, scene caption) — no hard-coded
object locations or names.

Usage
-----
  # Single instruction:
  python mujoco-cli-vision.py \\
      --mujoco-cli /path/to/mujoco-cli \\
      "Pick up the red cube"

  # Interactive mode (UNDO / CLEAR supported):
  python mujoco-cli-vision.py \\
      --mujoco-cli /path/to/mujoco-cli \\
      --interactive

  # Use lighter model:
  python mujoco-cli-vision.py \\
      --mujoco-cli /path/to/mujoco-cli \\
      --model microsoft/Florence-2-base \\
      "Stack the cubes"

All mujoco-cli flags (--scene, --seed, --max-retries, --no-viewer,
--output, --fps) are forwarded unchanged.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from PIL import Image

from vision.analyzer import SceneAnalyzer


# ---------------------------------------------------------------------------
# Dynamic loader for mujoco-cli
# ---------------------------------------------------------------------------

def _bootstrap_mujoco_cli(mujoco_cli_path: str):
    """
    Add the mujoco-cli directory to sys.path and return the loaded module.

    Accepts either a path to the directory (containing mujoco-cli.py and
    src/) or a direct path to mujoco-cli.py itself.
    """
    p = Path(mujoco_cli_path)
    cli_dir = p.parent if p.is_file() else p
    cli_script = cli_dir / "mujoco-cli.py"

    if not cli_script.exists():
        raise FileNotFoundError(
            f"mujoco-cli.py not found in '{cli_dir}'.\n"
            "Pass the directory that contains mujoco-cli.py via --mujoco-cli."
        )

    # Put mujoco-cli's directory first so `from src import ...` resolves
    sys.path.insert(0, str(cli_dir))

    spec = importlib.util.spec_from_file_location("mujoco_cli", cli_script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mujoco_cli"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Vision-based describe_scene
# ---------------------------------------------------------------------------

def _build_vision_describe_scene(analyzer: SceneAnalyzer):
    """
    Return a drop-in replacement for src.agent.describe_scene that uses
    Florence-2 to perceive the scene visually.

    The returned function renders the current MuJoCo frame, runs it through
    Florence-2 (<DETAILED_CAPTION> + <OD>), and produces a text block
    suitable for inclusion in the planner prompt.

    The robot's own joint/gripper state is still read from the environment
    (it is not a scene property — the controller always knows its own pose).
    """
    def describe_scene(env) -> str:
        # Render current frame and analyse with Florence-2
        frame = env.render()                       # numpy uint8 RGB (H, W, 3)
        image = Image.fromarray(frame).convert("RGB")
        scene = analyzer.analyze_scene(image)

        # Robot state (joint / gripper — not visual, always available)
        obs = env.get_obs()
        ee = obs["ee_pos"].round(3).tolist()
        qpos = obs["qpos"].round(3).tolist()
        grip = float(obs["finger_pos"].mean())
        if grip > 0.035:
            grip_desc = "fully open"
        elif grip < 0.005:
            grip_desc = "fully closed"
        else:
            grip_desc = "partially closed"

        robot_block = "\n".join([
            "=== Robot State ===",
            f"End-effector position: {ee}",
            f"Arm joints (rad): {qpos}",
            f"Gripper: {grip:.3f} ({grip_desc}; 0.04=open, 0=closed;"
            " ~0.01-0.03 when holding an object)",
            "Table: centre at (0.55, 0, 0), surface at z=0.55",
            "Drop zone: (0.55, 0.0, 0.58)",
        ])

        return scene.to_scene_text() + "\n\n" + robot_block

    return describe_scene


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_vision_args():
    """
    Parse the vision-specific flags from sys.argv.

    Returns (mujoco_cli_path, model_id, device, forwarded_argv).
    We cannot use argparse.parse_known_args here without special care
    because mujoco-cli also has positional args that argparse may steal,
    so we do a simple manual scan.
    """
    argv = sys.argv[1:]
    mujoco_cli_path = os.environ.get("MUJOCO_CLI_PATH", "")
    model_id = "microsoft/Florence-2-large"
    device = "auto"
    forwarded = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--mujoco-cli",) and i + 1 < len(argv):
            mujoco_cli_path = argv[i + 1]
            i += 2
        elif arg.startswith("--mujoco-cli="):
            mujoco_cli_path = arg.split("=", 1)[1]
            i += 1
        elif arg in ("--model",) and i + 1 < len(argv):
            model_id = argv[i + 1]
            i += 2
        elif arg.startswith("--model="):
            model_id = arg.split("=", 1)[1]
            i += 1
        elif arg in ("--device",) and i + 1 < len(argv):
            device = argv[i + 1]
            i += 2
        elif arg.startswith("--device="):
            device = arg.split("=", 1)[1]
            i += 1
        elif arg in ("-h", "--help"):
            _print_help()
            sys.exit(0)
        else:
            forwarded.append(arg)
            i += 1
    return mujoco_cli_path, model_id, device, forwarded


def _print_help():
    print(__doc__)
    print("Vision flags (consumed before forwarding to mujoco-cli):")
    print("  --mujoco-cli PATH   Path to mujoco-cli dir or mujoco-cli.py")
    print("                      (or set MUJOCO_CLI_PATH env var)")
    print("  --model MODEL_ID    Florence-2 model (default: microsoft/Florence-2-large)")
    print("  --device DEVICE     Torch device: auto / cpu / cuda / mps (default: auto)")
    print()
    print("All other flags are forwarded to mujoco-cli.py (--scene, --seed,")
    print("--max-retries, --no-viewer, --output, --fps, --interactive, ...).")


def main():
    mujoco_cli_path, model_id, device, forwarded_argv = _parse_vision_args()

    if not mujoco_cli_path:
        print(
            "Error: specify the mujoco-cli directory via --mujoco-cli PATH\n"
            "       or set the MUJOCO_CLI_PATH environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load Florence-2 once, in-process — no server required
    print(f"\U0001f50d Loading Florence-2 ({model_id}) on {device}\u2026", flush=True)
    analyzer = SceneAnalyzer(model_name=model_id, device=device)
    print("\u2705 Vision model ready.\n", flush=True)

    # Load mujoco-cli and its src/ package
    mujoco_cli_mod = _bootstrap_mujoco_cli(mujoco_cli_path)

    # Patch describe_scene in src.agent with the vision version
    from src import agent as _agent  # noqa: E402 (available after _bootstrap)
    _agent.describe_scene = _build_vision_describe_scene(analyzer)

    # Forward the remaining argv to mujoco-cli.main()
    sys.argv = [sys.argv[0]] + forwarded_argv
    mujoco_cli_mod.main()


if __name__ == "__main__":
    main()
