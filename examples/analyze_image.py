#!/usr/bin/env python3
"""
examples/analyze_image.py
=========================
Standalone demo: analyse any image with Florence-2 and print the scene context.

Usage
-----
    python examples/analyze_image.py path/to/image.png
    python examples/analyze_image.py path/to/image.png --dense
    python examples/analyze_image.py path/to/image.png --query "red cube. blue sphere"
    python examples/analyze_image.py path/to/image.png --model microsoft/Florence-2-base
"""

import argparse
import sys
from pathlib import Path

# Make the project root importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision.analyzer import SceneAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyse a scene image with Florence-2."
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument(
        "--model",
        default=SceneAnalyzer.DEFAULT_MODEL,
        help="Florence-2 model ID (default: %(default)s)",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Use dense region captioning (slower, richer output)",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help='Open-vocabulary query, e.g. "red cube. robot gripper"',
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device (default: auto)",
    )
    args = parser.parse_args()

    print(f"Loading Florence-2 ({args.model}) on {args.device} …")
    analyzer = SceneAnalyzer(model_name=args.model, device=args.device)

    image = SceneAnalyzer.load_image(args.image)
    print(f"Image: {image.width}×{image.height}px\n")

    if args.query:
        print(f"Query: "{args.query}"\n")
        detections = analyzer.ground_objects(image, args.query)
        if detections:
            for i, d in enumerate(detections, 1):
                cx, cy = d.center_norm
                print(f"  {i}. {d.label}  centre ({cx:.3f}, {cy:.3f})  "
                      f"area {d.area_norm*100:.1f}%")
        else:
            print("  No matches found.")
    else:
        analysis = analyzer.analyze_scene(image, use_dense_captions=args.dense)
        print(analysis.to_context_string())


if __name__ == "__main__":
    main()
