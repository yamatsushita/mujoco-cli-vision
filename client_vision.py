#!/usr/bin/env python3
"""
client_vision.py
================
Extended Remote CLI client with automatic scene perception.

Drop-in replacement for ``mujoco-cli.py`` that adds:

  1. **Automatic scene injection** — before every Copilot prompt, the client
     calls the vision server to retrieve a scene description and prepends it
     to the prompt.  The Copilot model therefore "knows" what objects are in
     the scene and where they are, even though no prior knowledge was given.

  2. **New built-in commands** (all start with ``\\``):

     | Command              | Description                                      |
     |----------------------|--------------------------------------------------|
     | ``\\scene``          | Show the current cached scene description        |
     | ``\\capture [cam]``  | Trigger a live MuJoCo render + analyse it        |
     | ``\\analyze <path>`` | Analyse an image file and cache the result       |
     | ``\\query <text>``   | Find specific objects in the cached scene        |
     | ``\\vision off/on``  | Temporarily disable/enable scene injection       |

  3. All standard ``mujoco-cli.py`` commands (``\ping``, ``\shell``, ``\clear``, etc.) are

Usage
-----
    # Start the vision server first (separate terminal):
    python -m vision.server --port 8765 --xml /path/to/model.xml

    # Then start the vision client:
    python client_vision.py \\
        --token ghp_xxx \\
        --name desktop \\
        --vision-url http://localhost:8765

Architecture
------------

    ┌────────────────────────────────────────────────────────────┐
    │  GitHub Issue (browser)                                    │
    │    User posts: "move the cube to the right of the sphere"  │
    └─────────────────────────┬──────────────────────────────────┘
                              │ poll
    ┌─────────────────────────▼──────────────────────────────────┐
    │  VisionCLIClient                                           │
    │    1. GET /scene  →  vision server                         │
    │    2. Build augmented prompt:                              │
    │         [scene context block]                              │
    │         ---                                                │
    │         User request: move the cube …                      │
    │    3. gh copilot -p "<augmented prompt>"                   │
    └─────────────────────────┬──────────────────────────────────┘
                              │ execute
    ┌─────────────────────────▼──────────────────────────────────┐
    │  MuJoCo simulation (local)                                 │
    └────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Optional

import requests

# ── Import the base client ────────────────────────────────────────────────────
# We add the parent directory so this file can coexist as a sibling of
# mujoco-cli.py inside the mujoco-cli workspace, or be run from anywhere if
# mujoco-cli.py is on the Python path.
_SCRIPT_DIR = Path(__file__).parent
_PARENT_DIR = _SCRIPT_DIR.parent

# Try to import RemoteCLIClient from different locations:
#   1. mujoco-cli.py in the same directory (standalone copy)
#   2. mujoco-cli.py in the parent repo root
_client_module = None
for _candidate in [_SCRIPT_DIR / "mujoco-cli.py", _PARENT_DIR / "mujoco-cli.py"]:
    if _candidate.exists():
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("mujoco_cli", str(_candidate))
        _client_module = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_client_module)
        break

if _client_module is None:
    raise ImportError(
        "Could not find mujoco-cli.py.  Place this file next to mujoco-cli.py "
        "(mujoco-cli root) or copy mujoco-cli.py to this directory."
    )

RemoteCLIClient = _client_module.RemoteCLIClient
_detect_repo_from_git = _client_module._detect_repo_from_git

logger = logging.getLogger(__name__)


# ── Vision-aware client ───────────────────────────────────────────────────────

class VisionCLIClient(RemoteCLIClient):
    """
    Extends RemoteCLIClient with automatic scene perception via a vision server.

    Parameters
    ----------
    vision_url:
        Base URL of the running vision server (default: http://localhost:8765).
    vision_enabled:
        Whether to automatically inject scene context into Copilot prompts.
    vision_timeout:
        HTTP timeout (seconds) when contacting the vision server.
    """

    def __init__(
        self,
        *args,
        vision_url: str = "http://localhost:8765",
        vision_enabled: bool = True,
        vision_timeout: float = 10.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vision_url = vision_url.rstrip("/")
        self.vision_enabled = vision_enabled
        self.vision_timeout = vision_timeout

    # ── Vision helpers ────────────────────────────────────────────────────────

    def _vision_get(self, path: str) -> Optional[dict]:
        try:
            resp = requests.get(
                f"{self.vision_url}{path}", timeout=self.vision_timeout
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.debug("Vision GET %s failed: %s", path, exc)
            return None

    def _vision_post(
        self,
        path: str,
        params: Optional[dict] = None,
        files: Optional[dict] = None,
        json_body: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> Optional[dict]:
        try:
            resp = requests.post(
                f"{self.vision_url}{path}",
                params=params,
                files=files,
                json=json_body,
                timeout=timeout or self.vision_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.debug("Vision POST %s failed: %s", path, exc)
            return None

    def _get_scene_context(self) -> Optional[str]:
        """Fetch the current scene context string from the vision server."""
        data = self._vision_get("/scene")
        if data:
            return data.get("context")
        return None

    def _analyze_file(self, path: str, dense: bool = False) -> Optional[dict]:
        """Upload a local image file to the vision server for analysis."""
        try:
            with open(path, "rb") as fh:
                data = self._vision_post(
                    "/analyze",
                    params={"dense": str(dense).lower()},
                    files={"file": fh},
                    timeout=120.0,
                )
            return data
        except FileNotFoundError:
            return None

    def _capture_scene(
        self,
        camera: str = "0",
        width: int = 640,
        height: int = 480,
        with_depth: bool = False,
    ) -> Optional[dict]:
        """Ask the vision server to render and analyse the current MuJoCo scene."""
        return self._vision_post(
            "/capture",
            params={
                "camera": camera,
                "width": width,
                "height": height,
                "with_depth": str(with_depth).lower(),
            },
            timeout=120.0,
        )

    def _query_objects(self, text: str) -> Optional[dict]:
        """Ask the vision server to find specific objects by description."""
        return self._vision_post("/query", params={"text": text}, timeout=60.0)

    # ── Augment Copilot prompts with scene context ─────────────────────────────

    def _run_copilot(self, prompt: str) -> Optional[str]:
        """
        Prepend the current scene context to the prompt before sending to
        Copilot CLI so the model knows the scene without prior knowledge.
        """
        if self.vision_enabled:
            ctx = self._get_scene_context()
            if ctx:
                prompt = f"{ctx}\n**User request:** {prompt}"
            else:
                prompt = (
                    "_(Vision server is offline or no scene has been captured "
                    "yet. Run `\\capture` to take a snapshot of the scene.)_\n\n"
                    f"**User request:** {prompt}"
                )
        return super()._run_copilot(prompt)

    # ── Override process_prompt to add vision commands ─────────────────────────

    def process_prompt(self, prompt: dict) -> Optional[str]:
        text = prompt["text"].strip()
        lower = text.lower()

        # ── \\scene ───────────────────────────────────────────────────────────
        if lower == "\\scene":
            data = self._vision_get("/scene")
            if data:
                return (
                    f"📷 **Current scene** (from vision server):\n\n"
                    f"{data['context']}"
                )
            return (
                "⚠️ No scene cached yet.\n"
                "Use `\\capture` to render the MuJoCo scene, "
                "or `\\analyze <image_path>` to analyse an image file."
            )

        # ── \\capture [camera] ────────────────────────────────────────────────
        if lower == "\\capture" or lower.startswith("\\capture "):
            camera = text[9:].strip() if lower.startswith("\\capture ") else "0"
            result = self._capture_scene(camera=camera or "0")
            if result:
                obj_count = len(result.get("scene", {}).get("objects", []))
                return (
                    f"📸 Scene captured and analysed ({obj_count} objects).\n\n"
                    f"{result['context']}"
                )
            return (
                "❌ Capture failed. Make sure the vision server is running "
                "with `--xml <model.xml>` and MuJoCo is accessible."
            )

        # ── \\analyze <path> ──────────────────────────────────────────────────
        if lower.startswith("\\analyze "):
            path = text[9:].strip()
            result = self._analyze_file(path)
            if result:
                obj_count = len(result.get("scene", {}).get("objects", []))
                return (
                    f"🔍 Image analysed ({obj_count} objects detected).\n\n"
                    f"{result['context']}"
                )
            return f"❌ Failed to analyse `{path}`. Check the path and that the vision server is running."

        # ── \\query <text> ────────────────────────────────────────────────────
        if lower.startswith("\\query "):
            query_text = text[7:].strip()
            result = self._query_objects(query_text)
            if result:
                matches = result.get("matches", [])
                if matches:
                    lines = [f"🎯 Found {len(matches)} match(es) for `{query_text}`:\n"]
                    for i, m in enumerate(matches, 1):
                        cx, cy = m.get("center_norm", [None, None])
                        xyz = m.get("world_xyz")
                        pos_str = (
                            f"world ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})"
                            if xyz
                            else f"image ({cx:.2f}, {cy:.2f})"
                        )
                        lines.append(f"  {i}. `{m['label']}` — {pos_str}")
                    return "\n".join(lines)
                return f"🔍 No matches found for `{query_text}`."
            return "❌ Query failed. Make sure the vision server is running."

        # ── \\vision on/off ───────────────────────────────────────────────────
        if lower in ("\\vision on", "\\vision off"):
            self.vision_enabled = lower.endswith("on")
            state = "enabled ✅" if self.vision_enabled else "disabled ⛔"
            return f"👁️ Scene injection {state}."

        # ── \\help override — add vision commands ─────────────────────────────
        if lower == "\\help":
            base_help = super().process_prompt(prompt) or ""
            vision_section = (
                "\n**Vision commands** (require vision server on "
                f"`{self.vision_url}`):\n\n"
                "| Command | Description |\n"
                "|---------|-------------|\n"
                "| `\\scene` | Show the current cached scene description |\n"
                "| `\\capture [cam]` | Render MuJoCo scene and analyse it |\n"
                "| `\\analyze <path>` | Analyse a local image file |\n"
                "| `\\query <text>` | Find specific objects in the scene |\n"
                "| `\\vision on/off` | Enable/disable auto scene injection |\n"
            )
            return base_help + vision_section

        return super().process_prompt(prompt)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vision-aware Remote CLI Client – MuJoCo + Florence-2"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub PAT (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument("--owner", help="Repository owner")
    parser.add_argument("--repo", help="Sessions repository name")
    parser.add_argument(
        "--name",
        default=platform.node(),
        help="Client name for multi-client routing",
    )
    parser.add_argument(
        "--vision-url",
        default=os.environ.get("VISION_URL", "http://localhost:8765"),
        help="Vision server base URL (default: http://localhost:8765)",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Start with scene injection disabled",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--new", action="store_true", help="Create a new session")
    group.add_argument("--join", type=int, metavar="N", help="Join issue #N")
    group.add_argument("--latest", action="store_true", help="Join the latest session")
    args = parser.parse_args()

    if not args.token:
        parser.error("Provide --token or set GITHUB_TOKEN.")

    owner, repo = args.owner, args.repo
    if not owner or not repo:
        detected = _detect_repo_from_git()
        if detected:
            owner = owner or detected[0]
            repo = repo or detected[1]
        else:
            parser.error("Could not auto-detect repo.  Pass --owner and --repo.")

    client = VisionCLIClient(
        token=args.token,
        owner=owner,
        repo=repo,
        name=args.name,
        vision_url=args.vision_url,
        vision_enabled=not args.no_vision,
    )

    # Print vision server status
    health = client._vision_get("/health")
    if health:
        model_ok = health.get("model_loaded", False)
        cap_ok = health.get("capture_available", False)
        print(
            f"👁️  Vision server at {args.vision_url}  "
            f"[model={'✅' if model_ok else '⏳'}  "
            f"capture={'✅' if cap_ok else '⚠️ (no XML)'}]"
        )
    else:
        print(
            f"⚠️  Vision server not reachable at {args.vision_url}.  "
            "Start it with: python -m vision.server"
        )
        if not args.no_vision:
            print("   Scene injection will be skipped until the server is available.")

    # Session management (identical to base client)
    if args.join:
        client.join_session(args.join)
    elif args.new:
        client.create_session()
    elif args.latest:
        n = client.find_latest_session()
        if n:
            client.join_session(n)
        else:
            client.create_session()
    else:
        n = client.find_own_session()
        if n:
            if client.is_name_active():
                parser.error(
                    f"Client '{args.name}' is already connected.  "
                    "Use --new, --join, or --latest."
                )
            client.join_session(n)
        else:
            client.create_session()

    client.run()


if __name__ == "__main__":
    main()
