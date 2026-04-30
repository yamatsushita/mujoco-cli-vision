#!/usr/bin/env python3
"""
mujoco-cli-vision.py
====================
Vision-augmented drop-in replacement for mujoco-cli.py.

Uses **RGB+D** images from the MuJoCo renderer to perceive the scene
without any pre-defined knowledge of object names or positions:

  RGB  → Florence-2   → object labels, bounding boxes, scene caption
  RGB+D → FoundationPose / depth fallback → 6-DoF object poses (position + orientation)

The combined perception is fed to the Copilot CLI planner which produces
a sequence of robot actions.  Those actions are executed and visualised
inside the MuJoCo simulation.

A dedicated ``vision_cam`` camera (injected into the scene XML) observes
the entire workspace from a slightly elevated 3/4 angle, providing both
good visual coverage and depth contrast for pose estimation.

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

  # Disable FoundationPose (use depth fallback only):
  python mujoco-cli-vision.py \\
      --mujoco-cli /path/to/mujoco-cli \\
      --no-foundation-pose \\
      "Pick up the red cube"

All mujoco-cli flags (--scene, --seed, --max-retries, --no-viewer,
--output, --fps) are forwarded unchanged.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from vision.analyzer import SceneAnalyzer
from vision.capture import MuJoCoCapture, VISION_CAM_POS, VISION_CAM_LOOKAT, VISION_CAM_FOVY
from vision.pose_estimator import PoseEstimator, ObjectPose


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
# Relax Panda joint limits for wider workspace
# ---------------------------------------------------------------------------

# Expanded joint ranges (radians).  Only joints that are too restrictive
# for general manipulation are widened; the rest keep their defaults.
_RELAXED_JOINT_RANGES = {
    'joint2': (-2.50,  2.50),    # shoulder: ±143° (was ±101°)
    'joint4': (-3.0718, 0.50),   # elbow: allow extension (was -3.07 to -0.07)
    'joint6': (-1.00,  3.7525),  # wrist: allow negative rotation (was -0.02 to 3.75)
}


def _patch_joint_limits():
    """
    Monkey-patch ``RobotEnv.__init__`` to relax joint limits and matching
    actuator ctrl ranges after the MuJoCo model is loaded.

    This gives the arm a wider reachable workspace — it can move the
    end-effector higher and perform side grasps more freely.
    """
    import mujoco as _mj
    from src.env import RobotEnv

    _original_init = RobotEnv.__init__

    def _relaxed_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        # Widen joint limits
        for jname, (lo, hi) in _RELAXED_JOINT_RANGES.items():
            jid = _mj.mj_name2id(self.model, _mj.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            self.model.jnt_range[jid] = [lo, hi]
            # Also widen the matching actuator ctrl range
            for aid in range(self.model.nu):
                ajid = self.model.actuator_trnid[aid, 0]
                if self.model.actuator_trntype[aid] == 0 and ajid == jid:
                    self.model.actuator_ctrlrange[aid] = [lo, hi]

    RobotEnv.__init__ = _relaxed_init


# ---------------------------------------------------------------------------
# Inject vision_cam into mujoco-cli's scene builder XML
# ---------------------------------------------------------------------------

_VISION_CAM_XML = (
    '    <camera name="vision_cam" '
    f'pos="{VISION_CAM_POS[0]:.2f} {VISION_CAM_POS[1]:.2f} {VISION_CAM_POS[2]:.2f}" '
    f'xyaxes="0.707 0.707 0 -0.28 0.28 0.92" fovy="{VISION_CAM_FOVY:.0f}"/>'
)


def _inject_vision_camera():
    """
    Patch mujoco-cli's scene_builder._SCENE_HEADER to include a vision_cam
    camera right after the existing front_cam definition.

    This is called once after mujoco-cli is bootstrapped, before any scene
    is loaded, so every generated XML will contain the perception camera.
    """
    try:
        from src import scene_builder as _sb
    except ImportError:
        return

    header = _sb._SCENE_HEADER
    if "vision_cam" in header:
        return  # already injected

    # Insert vision_cam after front_cam element (which may span multiple lines)
    marker = '<camera name="front_cam"'
    idx = header.find(marker)
    if idx < 0:
        return

    # Find the closing '/>' of the front_cam element
    close = header.find("/>", idx)
    if close < 0:
        return
    # Move past '/>' and the newline
    insert_pos = close + 2
    eol = header.find("\n", insert_pos)
    if eol >= 0:
        insert_pos = eol + 1

    _sb._SCENE_HEADER = header[:insert_pos] + "\n" + _VISION_CAM_XML + "\n" + header[insert_pos:]


# ---------------------------------------------------------------------------
# Perception cache — single perception per planning step
# ---------------------------------------------------------------------------

class PerceptionCache:
    """
    Caches the latest perception result so that ``describe_scene`` and
    ``_build_action_reference`` share the same data without redundant
    inference.
    """
    def __init__(
        self,
        analyzer: SceneAnalyzer,
        pose_estimator: PoseEstimator,
        cam_name: str = "vision_cam",
        width: int = 640,
        height: int = 480,
        output_rgbd: Optional[str] = None,
        show_rgbd: bool = False,
    ):
        self.analyzer = analyzer
        self.pose_estimator = pose_estimator
        self.cam_name = cam_name
        self.width = width
        self.height = height
        self.output_rgbd = output_rgbd
        self.show_rgbd = show_rgbd
        self._frame_counter = 0

        # Cached results
        self._caption: str = ""
        self._object_poses: list[ObjectPose] = []
        self._pose_mode: str = "unknown"
        self._capture: Optional[MuJoCoCapture] = None

    def perceive(self, env) -> None:
        """
        Run the full RGB+D perception pipeline on the current scene.

        1. Render RGB + depth from the vision_cam
        2. Florence-2: scene caption + object detection (from RGB)
        3. PoseEstimator: 6-DoF poses (from RGB+D + bboxes)
        """
        # Lazy-init the MuJoCoCapture wrapper on first use
        if self._capture is None:
            self._capture = MuJoCoCapture(model=env.model, data=env.data)

        # Determine which camera to use
        import mujoco
        cam = self.cam_name
        try:
            cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            if cam_id < 0:
                cam = "front_cam"
        except Exception:
            cam = "front_cam"

        # Capture RGB + Depth
        rgb_image, depth_map = self._capture.capture_with_depth(
            width=self.width, height=self.height, camera=cam,
        )

        # Save RGB + depth if --output_rgbd was specified
        if self.output_rgbd:
            self._save_rgbd(rgb_image, depth_map)

        # Display RGB + depth if --show_rgbd was specified
        if self.show_rgbd:
            self._display_rgbd(rgb_image, depth_map)

        # Florence-2: scene caption + object detection (RGB only)
        scene_analysis = self.analyzer.analyze_scene(rgb_image)
        self._caption = scene_analysis.caption

        # Get camera parameters for 3D unprojection
        camera_params = self._capture.get_camera_params(
            camera=cam, width=self.width, height=self.height,
        )

        # Convert detections to dicts for pose estimator
        det_dicts = [det.to_dict() for det in scene_analysis.objects]

        # PoseEstimator: 6-DoF poses from RGB+D
        rgb_array = np.array(rgb_image)
        self._object_poses = self.pose_estimator.estimate_poses(
            rgb=rgb_array,
            depth=depth_map,
            camera_params=camera_params,
            detections=det_dicts,
        )
        self._pose_mode = self.pose_estimator.mode

    def _save_rgbd(self, rgb_image: Image, depth_map: np.ndarray) -> None:
        """Save the captured RGB image, raw depth, and colourised depth visualisation."""
        out_dir = Path(self.output_rgbd)
        out_dir.mkdir(parents=True, exist_ok=True)

        idx = self._frame_counter
        self._frame_counter += 1

        # Save RGB as PNG
        rgb_path = out_dir / f"rgb_{idx:04d}.png"
        rgb_image.save(rgb_path)

        # Save depth as 16-bit PNG (millimetres) for lossless storage
        # and as .npy for exact float values
        depth_mm = (depth_map * 1000.0).astype(np.uint16)
        depth_raw_path = out_dir / f"depth_{idx:04d}.png"
        Image.fromarray(depth_mm).save(depth_raw_path)

        depth_npy_path = out_dir / f"depth_{idx:04d}.npy"
        np.save(depth_npy_path, depth_map)

        # Save colourised depth visualisation
        depth_vis_path = out_dir / f"depth_vis_{idx:04d}.png"
        self._save_depth_vis(depth_map, depth_vis_path)

        print(f"\U0001f4f7 Saved RGB+D frame {idx}: {rgb_path}, {depth_raw_path}, {depth_vis_path}",
              flush=True)

    @staticmethod
    def _save_depth_vis(depth_map: np.ndarray, path: Path) -> None:
        """Save a colourised depth map as an RGB PNG using the turbo colourmap."""
        try:
            import matplotlib.cm as cm
        except ImportError:
            # matplotlib unavailable — skip visualisation
            return

        valid = depth_map > 0
        if valid.any():
            vmin, vmax = depth_map[valid].min(), depth_map[valid].max()
        else:
            vmin, vmax = 0.0, 1.0

        # Normalise to [0, 1] and apply turbo colourmap
        rng = vmax - vmin if vmax > vmin else 1.0
        normed = np.clip((depth_map - vmin) / rng, 0.0, 1.0)
        normed[~valid] = 0.0
        coloured = (cm.turbo(normed)[:, :, :3] * 255).astype(np.uint8)
        # Mark invalid pixels as black
        coloured[~valid] = 0
        Image.fromarray(coloured).save(path)

    def _display_rgbd(self, rgb_image: Image, depth_map: np.ndarray) -> None:
        """Display the captured RGB image and depth map side by side."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\u26a0\ufe0f  matplotlib not installed — cannot display RGB+D. "
                  "Install with: pip install matplotlib", flush=True)
            self.show_rgbd = False  # disable for future calls
            return

        fig, (ax_rgb, ax_depth) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Vision Camera Observation (frame {self._frame_counter - 1})",
                     fontsize=13)

        ax_rgb.imshow(rgb_image)
        ax_rgb.set_title("RGB")
        ax_rgb.axis("off")

        # Colourised depth — mask out invalid (zero) pixels
        depth_vis = depth_map.copy()
        valid = depth_vis > 0
        if valid.any():
            vmin, vmax = depth_vis[valid].min(), depth_vis[valid].max()
        else:
            vmin, vmax = 0.0, 1.0
        im = ax_depth.imshow(depth_vis, cmap="turbo", vmin=vmin, vmax=vmax)
        ax_depth.set_title("Depth (metres)")
        ax_depth.axis("off")
        fig.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.04, label="m")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    @property
    def caption(self) -> str:
        return self._caption

    @property
    def object_poses(self) -> list[ObjectPose]:
        return self._object_poses

    @property
    def pose_mode(self) -> str:
        return self._pose_mode


# ---------------------------------------------------------------------------
# Vision-based describe_scene (RGB+D)
# ---------------------------------------------------------------------------

def _build_vision_describe_scene(cache: PerceptionCache):
    """
    Return a drop-in replacement for ``src.agent.describe_scene`` that uses
    the RGB+D perception pipeline.

    Runs Florence-2 (scene caption + object detection) and FoundationPose /
    depth-based pose estimation on every call, caching results so both
    ``describe_scene`` and ``_build_action_reference`` see the same data.
    """
    def describe_scene(env) -> str:
        cache.perceive(env)

        lines = [
            "=== Visual Scene (Florence-2 + pose estimation) ===",
            f"Perception mode: {cache.pose_mode}",
            f"Scene overview: {cache.caption}",
            "",
        ]

        if cache.object_poses:
            lines.append("Detected objects with estimated poses:")
            for i, pose in enumerate(cache.object_poses, start=1):
                pos_str = pose.position_str()
                orient_str = pose.orientation_str()
                mode_tag = "[6DoF]" if pose.pose_mode == "full" else "[pos]"
                conf_pct = pose.confidence * 100
                lines.append(
                    f"  {i:2d}. '{pose.label}' {mode_tag} "
                    f"position={pos_str}, orientation={orient_str}, "
                    f"conf={conf_pct:.0f}%"
                )
        else:
            lines.append("  (no objects detected)")

        lines += [
            "",
            "Coordinate frame: MuJoCo world frame (metres).",
            "Table: centre at (0.55, 0, 0), surface at z=0.55.",
            "Drop zone (neutral area): (0.55, 0.0, 0.58).",
        ]
        return "\n".join(lines)

    return describe_scene


def _build_vision_action_reference(cache: PerceptionCache):
    """
    Return a drop-in replacement for ``src.agent._build_action_reference``
    that derives all scene information from the cached RGB+D perception.
    """
    def action_reference(env) -> str:
        # If perception hasn't been run yet for this step, run it
        if not cache.object_poses and not cache.caption:
            cache.perceive(env)

        # Build object list from cached pose estimates
        obj_lines = []
        for pose in cache.object_poses:
            pos_str = pose.position_str()
            orient_str = pose.orientation_str()
            mode_tag = "[6DoF]" if pose.pose_mode == "full" else "[pos]"
            obj_lines.append(
                f"  - '{pose.label}' {mode_tag}: "
                f"world pos={pos_str}, orient={orient_str}"
            )

        if obj_lines:
            obj_block = (
                "Detected objects (from RGB+D perception):\n"
                + "\n".join(obj_lines)
            )
        else:
            obj_block = "No objects detected in the scene."

        # Robot action capabilities
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

        return (
            actions + "\n" + obj_block + "\n\n"
            "Table surface at z=0.55. Objects rest at z≈0.55+half_height. "
            "Drop zone: [0.55, 0.0, 0.58].\n"
            "Gripper: 0.04=fully open, 0=fully closed; "
            "fingers slip to ~0.01-0.03 when holding an object.\n"
            "Object positions are in MuJoCo world coordinates (metres).\n"
            "Use the detected positions to plan precise pick/place actions.\n"
        )

    return action_reference


# ---------------------------------------------------------------------------
# Fuzzy object name resolution
# ---------------------------------------------------------------------------

def _normalize(name: str) -> set[str]:
    """Split a name into lowercase keyword tokens for fuzzy matching."""
    import re
    return set(re.split(r'[\s_\-]+', name.lower().strip()))


def _fuzzy_match(query: str, candidates: list[str]) -> Optional[str]:
    """
    Find the best matching candidate for *query* using keyword overlap.

    Scoring:
      1. Exact match (case-insensitive, underscores ↔ spaces) → immediate win
      2. Token overlap score = |intersection| / |union|  (Jaccard similarity)
      3. Substring containment bonus

    Returns the best candidate, or None if no reasonable match is found.
    """
    q_norm = query.lower().replace(' ', '_').strip()
    q_tokens = _normalize(query)

    best, best_score = None, 0.0
    for cand in candidates:
        c_norm = cand.lower().replace(' ', '_').strip()

        # Exact match
        if q_norm == c_norm:
            return cand

        # Token overlap (Jaccard)
        c_tokens = _normalize(cand)
        inter = q_tokens & c_tokens
        union = q_tokens | c_tokens
        score = len(inter) / len(union) if union else 0.0

        # Substring bonus: if the query or candidate contains the other
        if q_norm in c_norm or c_norm in q_norm:
            score += 0.3

        # Colour word bonus: strong signal if colour matches
        colours = {'red', 'blue', 'green', 'yellow', 'orange', 'purple',
                   'white', 'black', 'pink', 'brown', 'gray', 'grey', 'cyan'}
        q_colours = q_tokens & colours
        c_colours = c_tokens & colours
        if q_colours and q_colours == c_colours:
            score += 0.3

        if score > best_score:
            best_score = score
            best = cand

    # Require at least some overlap to avoid nonsense matches
    return best if best_score >= 0.25 else None


def _patch_fuzzy_object_names():
    """
    Monkey-patch ``RobotController.get_object_pos`` so that object names
    from the LLM (e.g. "red object", "the blue block") are fuzzy-matched
    against the actual MuJoCo body names (e.g. "red_cube", "blue_cube").
    """
    from src.robot import RobotController

    _original_get_object_pos = RobotController.get_object_pos

    def _fuzzy_get_object_pos(self, obj_name):
        # Try exact match first
        result = _original_get_object_pos(self, obj_name)
        if result is not None:
            return result

        # Fuzzy match against known body names
        known = list(self.env.obj_body_ids.keys())
        match = _fuzzy_match(obj_name, known)
        if match is not None:
            print(f"  \U0001f50d Resolved '{obj_name}' → '{match}'")
            return _original_get_object_pos(self, match)

        return None

    RobotController.get_object_pos = _fuzzy_get_object_pos

    # Also patch _get_half_height, _get_obj_yaw, _activate_weld,
    # _check_grasp_alignment, _has_finger_contact_with, and _teleport_obj
    # which all take obj_name and look up MuJoCo bodies directly.
    _original_get_half_height = RobotController._get_half_height
    _original_get_obj_yaw = RobotController._get_obj_yaw
    _original_activate_weld = RobotController._activate_weld
    _original_has_finger_contact = RobotController._has_finger_contact_with
    _original_check_alignment = RobotController._check_grasp_alignment
    _original_teleport = RobotController._teleport_obj

    def _resolve(self, obj_name):
        """Resolve obj_name to the closest known body name."""
        if obj_name in self.env.obj_body_ids:
            return obj_name
        known = list(self.env.obj_body_ids.keys())
        match = _fuzzy_match(obj_name, known)
        return match if match is not None else obj_name

    def _fuzzy_get_half_height(self, obj_name):
        return _original_get_half_height(self, _resolve(self, obj_name))

    def _fuzzy_get_obj_yaw(self, obj_name):
        return _original_get_obj_yaw(self, _resolve(self, obj_name))

    def _fuzzy_activate_weld(self, obj_name):
        return _original_activate_weld(self, _resolve(self, obj_name))

    def _fuzzy_has_finger_contact(self, obj_name):
        return _original_has_finger_contact(self, _resolve(self, obj_name))

    def _fuzzy_check_alignment(self, obj_name, min_dot=0.75):
        return _original_check_alignment(self, _resolve(self, obj_name), min_dot)

    def _fuzzy_teleport(self, obj_name, world_pos, world_quat=None):
        return _original_teleport(self, _resolve(self, obj_name), world_pos, world_quat)

    RobotController._get_half_height = _fuzzy_get_half_height
    RobotController._get_obj_yaw = _fuzzy_get_obj_yaw
    RobotController._activate_weld = _fuzzy_activate_weld
    RobotController._has_finger_contact_with = _fuzzy_has_finger_contact
    RobotController._check_grasp_alignment = _fuzzy_check_alignment
    RobotController._teleport_obj = _fuzzy_teleport


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_vision_args():
    """
    Parse the vision-specific flags from sys.argv.

    Returns (mujoco_cli_path, model_id, device, use_fp, output_rgbd, forwarded_argv).
    We cannot use argparse.parse_known_args here without special care
    because mujoco-cli also has positional args that argparse may steal,
    so we do a simple manual scan.
    """
    argv = sys.argv[1:]
    mujoco_cli_path = os.environ.get("MUJOCO_CLI_PATH", "")
    model_id = "microsoft/Florence-2-large"
    device = "auto"
    use_fp = True
    output_rgbd = ""
    show_rgbd = False
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
        elif arg == "--no-foundation-pose":
            use_fp = False
            i += 1
        elif arg in ("--output_rgbd",) and i + 1 < len(argv):
            output_rgbd = argv[i + 1]
            i += 2
        elif arg.startswith("--output_rgbd="):
            output_rgbd = arg.split("=", 1)[1]
            i += 1
        elif arg == "--show_rgbd":
            show_rgbd = True
            i += 1
        elif arg in ("-h", "--help"):
            _print_help()
            sys.exit(0)
        else:
            forwarded.append(arg)
            i += 1
    return mujoco_cli_path, model_id, device, use_fp, output_rgbd, show_rgbd, forwarded


def _print_help():
    print(__doc__)
    print("Vision flags (consumed before forwarding to mujoco-cli):")
    print("  --mujoco-cli PATH       Path to mujoco-cli dir or mujoco-cli.py")
    print("                          (or set MUJOCO_CLI_PATH env var)")
    print("  --model MODEL_ID        Florence-2 model (default: microsoft/Florence-2-large)")
    print("  --device DEVICE         Torch device: auto / cpu / cuda / mps (default: auto)")
    print("  --no-foundation-pose    Disable FoundationPose (use depth fallback only)")
    print("  --output_rgbd DIR       Save captured RGB + depth images to DIR")
    print("  --show_rgbd             Display RGB + depth map in a window")
    print()
    print("All other flags are forwarded to mujoco-cli.py (--scene, --seed,")
    print("--max-retries, --no-viewer, --output, --fps, --interactive, ...).")


def main():
    mujoco_cli_path, model_id, device, use_fp, output_rgbd, show_rgbd, forwarded_argv = _parse_vision_args()

    if not mujoco_cli_path:
        print(
            "Error: specify the mujoco-cli directory via --mujoco-cli PATH\n"
            "       or set the MUJOCO_CLI_PATH environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load vision models ────────────────────────────────────────────────
    print(f"\U0001f50d Loading Florence-2 ({model_id}) on {device}\u2026", flush=True)
    analyzer = SceneAnalyzer(model_name=model_id, device=device)
    print("\u2705 Florence-2 ready.", flush=True)

    print(f"\U0001f4d0 Initialising pose estimator "
          f"(FoundationPose={'enabled' if use_fp else 'disabled'})\u2026",
          flush=True)
    pose_estimator = PoseEstimator(use_foundation_pose=use_fp)
    print(f"\u2705 Pose estimator ready (mode: {pose_estimator.mode}).\n", flush=True)

    # ── Load mujoco-cli and inject vision camera ──────────────────────────
    mujoco_cli_mod = _bootstrap_mujoco_cli(mujoco_cli_path)
    _inject_vision_camera()
    _patch_joint_limits()

    # ── Build perception cache ────────────────────────────────────────────
    cache = PerceptionCache(
        analyzer=analyzer,
        pose_estimator=pose_estimator,
        cam_name="vision_cam",
        output_rgbd=output_rgbd or None,
        show_rgbd=show_rgbd,
    )

    # ── Patch describe_scene and action_reference ─────────────────────────
    from src import agent as _agent  # noqa: E402 (available after _bootstrap)
    import src as _src

    vision_describe = _build_vision_describe_scene(cache)
    _agent.describe_scene = vision_describe
    _src.describe_scene = vision_describe
    mujoco_cli_mod.describe_scene = vision_describe

    _agent._build_action_reference = _build_vision_action_reference(cache)

    # ── Patch RobotController with fuzzy object name resolution ───────────
    _patch_fuzzy_object_names()

    # ── Forward to mujoco-cli ─────────────────────────────────────────────
    sys.argv = [sys.argv[0]] + forwarded_argv
    mujoco_cli_mod.main()


if __name__ == "__main__":
    main()
