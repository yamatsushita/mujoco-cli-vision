"""
vision/capture.py
=================
MuJoCo scene capture utility.

Renders the simulation from a fixed (or named) camera, returns RGB images and
optional depth maps, and provides helpers to project 2-D image coordinates
back into 3-D world space using the camera's intrinsic / extrinsic parameters
stored in the MuJoCo model.

Design notes
------------
* Works with any MuJoCo model that has at least one named or indexed camera.
* The ``Renderer`` is created lazily and cached; it is re-created only when
  the resolution changes.
* ``localize_3d`` uses the camera projection matrix (fovy, image size, extrinsics)
  to unproject a pixel + depth value into world coordinates.  This matches the
  standard pinhole model used by MuJoCo's fixed cameras.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# mujoco is an optional dependency; importing it here so the module can be
# imported without it (analyzer-only usage), but capture will fail gracefully.
try:
    import mujoco
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False
    logger.warning(
        "mujoco package not found. MuJoCoCapture will not be usable."
    )


# Default perception camera parameters: positioned to see the entire
# Franka Panda workspace (arm + table + objects) from a slightly elevated
# 3/4 angle.  Provides good depth contrast for pose estimation.
VISION_CAM_POS = np.array([1.3, -0.75, 1.55])
VISION_CAM_LOOKAT = np.array([0.50, 0.0, 0.55])
VISION_CAM_FOVY = 52.0  # degrees


def inject_vision_camera(model, data, cam_name: str = "vision_cam") -> int:
    """
    Check whether a named camera exists in the model.  If it does, return
    its ID.  MuJoCo does not support adding cameras at runtime, so the
    camera must be defined in the XML.  This function simply resolves the
    name to an ID and logs a warning if it is missing.

    Returns the camera ID, or -1 if not found.
    """
    try:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    except Exception:
        cam_id = -1
    if cam_id < 0:
        logger.warning(
            "Camera '%s' not found in model.  "
            "Using camera index 0 (front_cam) as fallback.",
            cam_name,
        )
    return cam_id


class MuJoCoCapture:
    """
    Renders MuJoCo scenes and converts 2-D detections to 3-D world coords.

    Parameters
    ----------
    xml_path:
        Path to a ``.xml`` model file.  Mutually exclusive with ``model``.
    model:
        Pre-loaded ``mujoco.MjModel``.  Mutually exclusive with ``xml_path``.
    data:
        ``mujoco.MjData`` to use.  Created automatically when ``xml_path`` is
        given or when ``model`` is given without ``data``.
    """

    def __init__(
        self,
        xml_path: Optional[Union[str, Path]] = None,
        model=None,
        data=None,
    ):
        if not _MUJOCO_AVAILABLE:
            raise ImportError("Install mujoco: pip install mujoco")

        if xml_path is not None:
            self.model = mujoco.MjModel.from_xml_path(str(xml_path))
            self.data = mujoco.MjData(self.model)
        elif model is not None:
            self.model = model
            self.data = data if data is not None else mujoco.MjData(model)
        else:
            raise ValueError("Provide either xml_path or model.")

        self._renderer: Optional[object] = None  # mujoco.Renderer
        self._renderer_size: tuple[int, int] = (0, 0)

    # ── Renderer lifecycle ────────────────────────────────────────────────────

    def _get_renderer(self, width: int, height: int):
        if self._renderer is None or self._renderer_size != (width, height):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderer_size = (width, height)
        return self._renderer

    def close(self):
        """Release the renderer resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Capture ───────────────────────────────────────────────────────────────

    def capture(
        self,
        width: int = 640,
        height: int = 480,
        camera: Union[int, str] = 0,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Image.Image:
        """
        Render the current simulation state as an RGB PIL Image.

        Parameters
        ----------
        camera:
            Integer camera index **or** name string matching a camera defined
            in the MuJoCo XML (``<camera name="fixed_cam" …/>``).
        output_path:
            If provided, save the image here (PNG format inferred from suffix).
        """
        renderer = self._get_renderer(width, height)
        renderer.update_scene(self.data, camera=camera)
        pixels: np.ndarray = renderer.render()          # uint8 [H, W, 3]
        image = Image.fromarray(pixels, mode="RGB")
        if output_path is not None:
            image.save(output_path)
            logger.debug("Scene image saved to %s", output_path)
        return image

    def capture_with_depth(
        self,
        width: int = 640,
        height: int = 480,
        camera: Union[int, str] = 0,
    ) -> tuple[Image.Image, np.ndarray]:
        """
        Render both RGB and depth map.

        Returns
        -------
        image:
            RGB PIL Image.
        depth:
            Float32 ndarray [H, W] of distances in metres (MuJoCo convention:
            values are in the ``[near, far]`` range set for the camera).
        """
        renderer = self._get_renderer(width, height)

        # RGB pass
        renderer.update_scene(self.data, camera=camera)
        rgb = renderer.render().copy()

        # Depth pass
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=camera)
        depth: np.ndarray = renderer.render().copy()
        renderer.disable_depth_rendering()

        return Image.fromarray(rgb, mode="RGB"), depth

    # ── 3-D localisation ──────────────────────────────────────────────────────

    def get_camera_params(
        self,
        camera: Union[int, str],
        width: int,
        height: int,
    ) -> dict:
        """
        Return the intrinsic / extrinsic parameters for a named camera.

        The pinhole model used by MuJoCo:
          f = 0.5 * height / tan(fovy / 2)   (in pixels)

        Returns a dict with keys:
          fovy_rad, f, cx, cy,
          position (3,), rotation_matrix (3, 3)
        """
        # Resolve camera index
        cam_id = (
            camera
            if isinstance(camera, int)
            else mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        )
        fovy_rad = self.model.cam_fovy[cam_id] * np.pi / 180.0
        f = 0.5 * height / np.tan(fovy_rad / 2.0)
        cx, cy = width / 2.0, height / 2.0

        pos = self.data.cam_xpos[cam_id].copy()          # (3,)
        rot = self.data.cam_xmat[cam_id].reshape(3, 3).copy()  # (3, 3)

        return {
            "fovy_rad": fovy_rad,
            "f": f,
            "cx": cx,
            "cy": cy,
            "position": pos,
            "rotation_matrix": rot,
        }

    def unproject(
        self,
        u: float,
        v: float,
        depth: float,
        camera_params: dict,
    ) -> np.ndarray:
        """
        Back-project a pixel (u, v) at known ``depth`` to 3-D world coordinates.

        Parameters
        ----------
        u, v:
            Pixel coordinates (origin = top-left, v increases downward).
        depth:
            Distance along the optical axis in metres.
        camera_params:
            Output of ``get_camera_params``.

        Returns
        -------
        world_xyz:
            ndarray of shape (3,) in world coordinates.
        """
        f  = camera_params["f"]
        cx = camera_params["cx"]
        cy = camera_params["cy"]
        pos = camera_params["position"]
        R   = camera_params["rotation_matrix"]

        # Camera-space direction (normalised)
        x_cam = (u - cx) / f
        y_cam = -(v - cy) / f   # flip y: image v↓ → camera y↑
        dir_cam = np.array([x_cam, y_cam, -1.0])  # MuJoCo: optical axis = -Z

        # Scale by depth along optical axis
        point_cam = dir_cam * depth / np.linalg.norm(dir_cam)

        # Transform to world frame: world = pos + R @ point_cam
        return pos + R @ point_cam

    def localize_objects_3d(
        self,
        detections: list[dict],
        depth_map: np.ndarray,
        camera: Union[int, str] = 0,
    ) -> list[dict]:
        """
        Augment each detection dict with a ``world_xyz`` field.

        Parameters
        ----------
        detections:
            List of dicts from ``SceneAnalyzer.detect_objects`` (or
            ``DetectedObject.to_dict()``).  Must contain ``"bbox_pixels"``.
        depth_map:
            Float32 ndarray [H, W] from ``capture_with_depth``.
        camera:
            Camera used when capturing.

        Returns
        -------
        Augmented detection dicts (new key ``"world_xyz": [x, y, z]``).
        """
        H, W = depth_map.shape[:2]
        params = self.get_camera_params(camera, width=W, height=H)
        results = []
        for det in detections:
            bbox = det.get("bbox_pixels") or det.get("bbox")
            if bbox is None:
                results.append({**det, "world_xyz": None})
                continue
            x1, y1, x2, y2 = bbox
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0
            iu, iv = min(int(round(u)), W - 1), min(int(round(v)), H - 1)
            d = float(depth_map[iv, iu])
            if d <= 0:
                results.append({**det, "world_xyz": None})
                continue
            world = self.unproject(u, v, d, params)
            results.append({**det, "world_xyz": world.tolist()})
        return results
