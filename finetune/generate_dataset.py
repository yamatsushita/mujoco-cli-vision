#!/usr/bin/env python3
"""
finetune/generate_dataset.py
=============================
Generate a fine-tuning dataset for Florence-2 object detection (<OD> task)
by rendering MuJoCo scenes with known object bounding boxes.

For each scene × seed combination the script:
  1. Creates a RobotEnv with the given scene/seed
  2. Renders the front-camera image (1280×720)
  3. Projects each object's 3-D position to 2-D pixel coordinates
  4. Computes a tight bounding box from the object's known size
  5. Saves the image + annotation in Florence-2 <OD> format

The output is a JSON-lines file (one sample per line) plus an images/
directory, suitable for direct consumption by the fine-tuning script.

Usage
-----
    python finetune/generate_dataset.py --mujoco-cli ../mujoco-cli
    python finetune/generate_dataset.py --mujoco-cli ../mujoco-cli --seeds 50
    python finetune/generate_dataset.py --mujoco-cli ../mujoco-cli --scenes 0 2 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _bootstrap_mujoco_cli(mujoco_cli_path: str):
    """Add mujoco-cli to sys.path and return the loaded module."""
    p = Path(mujoco_cli_path)
    cli_dir = p.parent if p.is_file() else p
    sys.path.insert(0, str(cli_dir))


def _project_to_pixel(
    world_pos: np.ndarray,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    fovy_rad: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Project a 3-D world point to 2-D pixel coordinates (u, v)."""
    # Camera frame: MuJoCo camera looks along -Z, with Y up
    rel = world_pos - cam_pos
    cam_coords = cam_mat.T @ rel  # (3,) in camera frame

    f = 0.5 * height / np.tan(fovy_rad / 2.0)
    cx, cy = width / 2.0, height / 2.0

    # Project: u = cx + f * x/(-z), v = cy - f * y/(-z)
    if cam_coords[2] >= 0:
        return cx, cy  # behind camera
    u = cx + f * cam_coords[0] / (-cam_coords[2])
    v = cy - f * cam_coords[1] / (-cam_coords[2])
    return float(u), float(v)


def _object_bbox_pixels(
    env,
    obj_name: str,
    obj_cfg: dict,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    fovy_rad: float,
    width: int,
    height: int,
) -> list[float] | None:
    """
    Compute a 2-D bounding box [x1, y1, x2, y2] in pixel coordinates
    for an object by projecting its 3-D corners.
    """
    import mujoco

    bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    pos = env.data.xpos[bid].copy()
    quat = env.data.xquat[bid].copy()  # [w, x, y, z]

    # Build rotation matrix from quaternion
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)

    # Get object half-extents in local frame
    shape = obj_cfg.get("shape", "box")
    size = obj_cfg.get("size", [0.025, 0.025, 0.025])

    if shape == "box":
        hx, hy, hz = size[0], size[1], size[2]
    elif shape == "cylinder":
        r, h = size[0], size[1]
        hx, hy, hz = r, r, h
    elif shape == "sphere":
        r = size[0]
        hx, hy, hz = r, r, r
    else:
        hx, hy, hz = 0.025, 0.025, 0.025

    # 8 corners of the local bounding box
    corners_local = np.array([
        [sx * hx, sy * hy, sz * hz]
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ])

    # Transform to world frame and project
    us, vs = [], []
    for corner in corners_local:
        world_corner = pos + rot @ corner
        u, v = _project_to_pixel(
            world_corner, cam_pos, cam_mat, fovy_rad, width, height
        )
        us.append(u)
        vs.append(v)

    x1 = max(0, min(us))
    y1 = max(0, min(vs))
    x2 = min(width, max(us))
    y2 = min(height, max(vs))

    # Skip if too small or off-screen
    if (x2 - x1) < 3 or (y2 - y1) < 3:
        return None

    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def _get_camera_params(env, camera_name: str = "front_cam"):
    """Extract camera intrinsic/extrinsic parameters from MuJoCo model."""
    import mujoco

    cam_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
    )
    fovy_rad = env.model.cam_fovy[cam_id] * np.pi / 180.0
    cam_pos = env.data.cam_xpos[cam_id].copy()
    cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3).copy()
    return cam_pos, cam_mat, fovy_rad


def _format_od_suffix(bboxes: list[list[float]], labels: list[str]) -> str:
    """
    Format bounding boxes and labels into Florence-2 <OD> response format.

    Florence-2 represents bounding boxes as special location tokens:
      <loc_XXX> where XXX is the quantised coordinate (0–999).
    """
    parts = []
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        parts.append(
            f"<loc_{int(x1)}><loc_{int(y1)}>"
            f"<loc_{int(x2)}><loc_{int(y2)}>"
            f"{label}"
        )
    return "".join(parts)


def _bbox_to_florence_quantised(
    bbox: list[float], img_w: int, img_h: int
) -> list[int]:
    """
    Convert pixel bounding box to Florence-2 quantised coordinates (0–999).
    """
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 / img_w * 999)),
        int(round(y1 / img_h * 999)),
        int(round(x2 / img_w * 999)),
        int(round(y2 / img_h * 999)),
    ]


def generate_dataset(
    mujoco_cli_path: str,
    output_dir: str,
    scenes: list[int] | None = None,
    n_seeds: int = 20,
    width: int = 1280,
    height: int = 720,
):
    """Generate the fine-tuning dataset."""
    _bootstrap_mujoco_cli(mujoco_cli_path)

    from src.env import RobotEnv
    from src.scenes import SCENE_CONFIGS, N_SCENES

    if scenes is None:
        scenes = list(range(N_SCENES))

    out_path = Path(output_dir)
    img_dir = out_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    total = 0

    for scene_id in scenes:
        scene_cfg = SCENE_CONFIGS[scene_id % N_SCENES]
        print(f"\nScene {scene_id}: {scene_cfg['name']}")
        print(f"  Objects: {[o['name'] for o in scene_cfg['objects']]}")

        for seed in range(n_seeds):
            env = RobotEnv(render_mode="offscreen", seed=seed, scene=scene_id)

            # Get camera parameters
            cam_pos, cam_mat, fovy_rad = _get_camera_params(env)

            # Render the scene
            frame = env.render(width=width, height=height)
            image = Image.fromarray(frame, mode="RGB")

            # Compute bounding boxes for each object
            bboxes = []
            labels = []
            for obj_cfg in scene_cfg["objects"]:
                name = obj_cfg["name"]
                bbox = _object_bbox_pixels(
                    env, name, obj_cfg,
                    cam_pos, cam_mat, fovy_rad,
                    width, height,
                )
                if bbox is not None:
                    bboxes.append(bbox)
                    labels.append(name)

            if not bboxes:
                continue

            # Save image
            img_name = f"scene{scene_id}_seed{seed:04d}.png"
            image.save(str(img_dir / img_name))

            # Convert bboxes to Florence-2 quantised format (0–999)
            q_bboxes = [
                _bbox_to_florence_quantised(b, width, height)
                for b in bboxes
            ]

            # Build Florence-2 OD suffix string
            suffix = _format_od_suffix(q_bboxes, labels)

            # Sample record
            sample = {
                "image": f"images/{img_name}",
                "prefix": "<OD>",
                "suffix": suffix,
                # Metadata (not used by training, useful for debugging)
                "scene_id": scene_id,
                "seed": seed,
                "objects": [
                    {"label": lbl, "bbox_pixels": bb, "bbox_quantised": qb}
                    for lbl, bb, qb in zip(labels, bboxes, q_bboxes)
                ],
            }
            samples.append(sample)
            total += 1

            if total % 10 == 0:
                print(f"  Generated {total} samples...", end="\r")

    # Write JSONL
    jsonl_path = out_path / "train.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n\nDataset generated:")
    print(f"  Samples: {total}")
    print(f"  Images:  {img_dir}")
    print(f"  Labels:  {jsonl_path}")

    # Also generate a caption dataset for <DETAILED_CAPTION> task
    caption_samples = []
    for s in samples:
        obj_names = [o["label"] for o in s["objects"]]
        caption = _generate_caption(s["scene_id"], obj_names)
        caption_samples.append({
            "image": s["image"],
            "prefix": "<DETAILED_CAPTION>",
            "suffix": caption,
        })

    caption_path = out_path / "train_caption.jsonl"
    with open(caption_path, "w", encoding="utf-8") as f:
        for s in caption_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Captions: {caption_path}")

    return samples


def _generate_caption(scene_id: int, obj_names: list[str]) -> str:
    """Generate a descriptive caption for a scene."""
    obj_list = ", ".join(obj_names[:-1]) + f" and {obj_names[-1]}" \
        if len(obj_names) > 1 else obj_names[0]

    # Convert underscored names to readable form
    readable = obj_list.replace("_", " ")

    templates = [
        f"A robotic arm is positioned above a table with {readable} on it.",
        f"The scene shows a robot arm and a table containing {readable}.",
        f"A Franka Panda robot arm hovers over a table with {readable} placed on the surface.",
    ]
    # Deterministic selection based on scene_id
    return templates[scene_id % len(templates)]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Florence-2 fine-tuning dataset from MuJoCo scenes"
    )
    parser.add_argument(
        "--mujoco-cli", required=True,
        help="Path to mujoco-cli directory",
    )
    parser.add_argument(
        "--output", default="finetune/data",
        help="Output directory (default: finetune/data)",
    )
    parser.add_argument(
        "--scenes", type=int, nargs="+", default=None,
        help="Scene IDs to include (default: all)",
    )
    parser.add_argument(
        "--seeds", type=int, default=20,
        help="Number of seeds per scene (default: 20)",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Render width (default: 1280)",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Render height (default: 720)",
    )
    args = parser.parse_args()

    generate_dataset(
        mujoco_cli_path=args.mujoco_cli,
        output_dir=args.output,
        scenes=args.scenes,
        n_seeds=args.seeds,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
