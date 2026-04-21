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
    Return a drop-in replacement for src.agent.describe_scene that relies
    entirely on Florence-2 vision analysis.

    The returned function renders the current MuJoCo frame, runs it through
    Florence-2 (<DETAILED_CAPTION> + <OD>), and returns the result as text.
    No simulation state is read — all information comes from the image.
    """
    def describe_scene(env) -> str:
        frame = env.render()
        image = Image.fromarray(frame).convert("RGB")
        scene = analyzer.analyze_scene(image)
        return scene.to_scene_text()

    return describe_scene


def _build_vision_action_reference(analyzer: SceneAnalyzer):
    """
    Return a drop-in replacement for src.agent._build_action_reference that
    derives all scene information from Florence-2 vision.

    No simulation configs, hardcoded coordinates, or hand-written scene
    descriptions are included.  The available robot actions are listed
    (these are the robot's capabilities, not scene assumptions), and
    object names are taken from Florence-2 detections.
    """
    def action_reference(env) -> str:
        # Use Florence-2 to detect objects visually
        frame = env.render()
        image = Image.fromarray(frame).convert("RGB")
        detections = analyzer.detect_objects(image)

        # Build object list from vision detections with shape cues
        obj_lines = []
        for det in detections:
            cx, cy = det.center_norm
            area_pct = det.area_norm * 100
            bw = det.bbox_pixels[2] - det.bbox_pixels[0]
            bh = det.bbox_pixels[3] - det.bbox_pixels[1]
            aspect = bw / bh if bh > 0 else 1.0

            shape_hint = ""
            if aspect > 1.5:
                shape_hint = " [wide/flat]"
            elif aspect < 0.67:
                shape_hint = " [tall]"

            obj_lines.append(
                f"  - '{det.label}' at image ({cx:.2f}, {cy:.2f}), "
                f"area {area_pct:.1f}%{shape_hint}"
            )

        if obj_lines:
            obj_block = "Visually detected objects:\n" + "\n".join(obj_lines)
        else:
            obj_block = "No objects detected in the scene."

        # Robot action capabilities (what the robot CAN do, not scene info)
        actions = (
            "Available HIGH-LEVEL ACTIONS:\n"
            "  pick_object(object_name) — grasp from the top\n"
            "  pick_object_side(object_name) — side grasp\n"
            "  place_at([x, y, z]) — place held object at coordinates\n"
            "  stack_on(object_name) — place held object on top of another\n"
            "  place_beside(object_name, side, gap) — place next to another; "
            "side: left|right|front|back\n"
            "  move_to([x, y, z]) — move end-effector to position\n"
            "  push_object(object_name, direction, distance) — slide object; "
            "direction: left|right|forward|back\n"
            "  sweep_to(object_name, target_pos) — push object to position\n"
            "  move_aside(object_name) — move object out of the way\n"
            "  go_home() — return to neutral pose\n"
            "  open_gripper() / close_gripper()\n"
            "  drop_object() — release held object\n"
            "  rotate_wrist(angle_rad) — rotate end-effector\n"
        )

        return actions + "\n" + obj_block + "\n"

    return action_reference


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

    # Patch describe_scene everywhere it was imported so ALL call sites
    # (including mujoco-cli.py's module-level binding) use the vision version.
    from src import agent as _agent  # noqa: E402 (available after _bootstrap)
    import src as _src
    vision_describe = _build_vision_describe_scene(analyzer)
    _agent.describe_scene = vision_describe
    _src.describe_scene = vision_describe
    mujoco_cli_mod.describe_scene = vision_describe

    # Patch _build_action_reference to use vision-derived object info
    _agent._build_action_reference = _build_vision_action_reference(analyzer)

    # Forward the remaining argv to mujoco-cli.main()
    sys.argv = [sys.argv[0]] + forwarded_argv
    mujoco_cli_mod.main()


if __name__ == "__main__":
    main()
