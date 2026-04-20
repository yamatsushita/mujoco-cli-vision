#!/usr/bin/env python3
"""
examples/mujoco_integration.py
===============================
Full end-to-end demo of the mujoco-cli-vision pipeline.

Demonstrates:
  1. Load a MuJoCo model (``cartpole.xml`` bundled with mujoco package)
  2. Render the scene from a fixed camera
  3. Analyse the image with Florence-2 (via vision server)
  4. Print the scene context that would be injected into the Copilot prompt

Prerequisites
-------------
    pip install mujoco
    # Start the vision server in a separate terminal:
    python -m vision.server --port 8765

Usage
-----
    python examples/mujoco_integration.py
    python examples/mujoco_integration.py --xml /path/to/your/model.xml
    python examples/mujoco_integration.py --no-server  # use Florence-2 directly
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from PIL import Image


def demo_with_server(xml_path: str, vision_url: str):
    """Trigger /capture on the vision server and display the result."""
    print(f"Vision server: {vision_url}")
    print(f"MuJoCo model:  {xml_path}\n")

    # Health check
    try:
        h = requests.get(f"{vision_url}/health", timeout=5).json()
        print(f"Server health: {h}")
    except Exception as e:
        print(f"⚠️  Cannot reach vision server: {e}")
        print("Start it with: python -m vision.server --xml <model.xml>")
        sys.exit(1)

    print("\nCapturing scene …")
    t0 = time.time()
    resp = requests.post(
        f"{vision_url}/capture",
        params={"camera": "0", "width": 640, "height": 480, "with_depth": "true"},
        timeout=120,
    )
    elapsed = time.time() - t0
    if resp.status_code != 200:
        print(f"Error: {resp.status_code} {resp.text}")
        sys.exit(1)

    data = resp.json()
    print(f"Done in {elapsed:.1f}s\n")
    print("─" * 60)
    print(data["context"])
    print("─" * 60)

    if "objects_3d" in data:
        print("\n3-D object positions (world frame):")
        for obj in data["objects_3d"]:
            xyz = obj.get("world_xyz")
            if xyz:
                print(f"  {obj['label']:30s}  xyz = ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})")


def demo_direct(xml_path: str):
    """Render and analyse directly without the HTTP server."""
    try:
        import mujoco
    except ImportError:
        print("mujoco not installed: pip install mujoco")
        sys.exit(1)

    from vision.analyzer import SceneAnalyzer
    from vision.capture import MuJoCoCapture

    print(f"MuJoCo model: {xml_path}\n")
    print("Loading Florence-2 …")
    analyzer = SceneAnalyzer()

    print("Rendering scene …")
    with MuJoCoCapture(xml_path=xml_path) as cap:
        image, depth = cap.capture_with_depth(width=640, height=480, camera=0)
        image.save("/tmp/mujoco_scene_demo.png")
        print("Saved to /tmp/mujoco_scene_demo.png")

        print("Analysing …")
        t0 = time.time()
        analysis = analyzer.analyze_scene(image)
        elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s\n")
    print("─" * 60)
    print(analysis.to_context_string())
    print("─" * 60)

    # 3-D positions
    with MuJoCoCapture(xml_path=xml_path) as cap:
        objects_3d = cap.localize_objects_3d(
            [o.to_dict() for o in analysis.objects], depth
        )

    print("\n3-D object positions (world frame):")
    for obj in objects_3d:
        xyz = obj.get("world_xyz")
        if xyz:
            print(
                f"  {obj['label']:30s}  xyz = "
                f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})"
            )


def main():
    # Find a default XML: try cartpole bundled with mujoco package
    try:
        import mujoco
        _dm_path = Path(mujoco.__file__).parent / "testdata" / "cartpole.xml"
        default_xml = str(_dm_path) if _dm_path.exists() else ""
    except ImportError:
        default_xml = ""

    parser = argparse.ArgumentParser(
        description="MuJoCo + vision-server end-to-end demo"
    )
    parser.add_argument(
        "--xml",
        default=default_xml,
        help="MuJoCo XML model path (default: cartpole.xml from mujoco package)",
    )
    parser.add_argument(
        "--vision-url",
        default="http://localhost:8765",
        help="Vision server URL",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Skip the HTTP server and run Florence-2 directly (slower startup)",
    )
    args = parser.parse_args()

    if not args.xml:
        print(
            "No XML path found.  Install mujoco or pass --xml /path/to/model.xml"
        )
        sys.exit(1)

    if args.no_server:
        demo_direct(args.xml)
    else:
        demo_with_server(args.xml, args.vision_url)


if __name__ == "__main__":
    main()
