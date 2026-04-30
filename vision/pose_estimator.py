"""
vision/pose_estimator.py
========================
6-DoF object pose estimation using **FoundationPose** (NVIDIA, MIT licence)
with a depth-based fallback for environments where FoundationPose is not
available.

FoundationPose estimates full 6-DoF poses (position + orientation) from
RGB-D images given either a CAD model or reference images of each object.
When FoundationPose is not installed, the estimator falls back to
depth-based unprojection (position only, no orientation).

Reference:
  https://github.com/NVlabs/FoundationPose
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Try importing FoundationPose ──────────────────────────────────────────────

try:
    import estimater as _fp_estimater  # FoundationPose top-level module
    _FP_AVAILABLE = True
except ImportError:
    _FP_AVAILABLE = False


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ObjectPose:
    """Estimated 6-DoF pose for a single detected object."""
    label: str
    position: list[float]           # [x, y, z] in world frame (metres)
    quaternion: list[float]         # [w, x, y, z] orientation quaternion
    euler_deg: list[float]          # [roll, pitch, yaw] in degrees
    confidence: float = 1.0
    pose_mode: str = "full"         # "full" (FoundationPose) or "position_only" (depth fallback)
    bbox_pixels: Optional[list[float]] = None  # [x1, y1, x2, y2] from detection

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "position": self.position,
            "quaternion": self.quaternion,
            "euler_deg": self.euler_deg,
            "confidence": self.confidence,
            "pose_mode": self.pose_mode,
            "bbox_pixels": self.bbox_pixels,
        }

    def position_str(self) -> str:
        x, y, z = self.position
        return f"({x:.3f}, {y:.3f}, {z:.3f})"

    def orientation_str(self) -> str:
        if self.pose_mode == "position_only":
            return "unknown"
        r, p, y = self.euler_deg
        return f"roll={r:.1f}° pitch={p:.1f}° yaw={y:.1f}°"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw] in degrees."""
    w, x, y, z = q
    # Roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # Yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees(np.array([roll, pitch, yaw]))


def _bbox_mask(depth: np.ndarray, bbox: list[float], margin: int = 2) -> np.ndarray:
    """
    Create a foreground mask within a bounding box using depth discontinuity.

    Selects pixels whose depth is within one standard deviation of the bbox
    median depth, filtering out background / table pixels that leak into the box.
    """
    H, W = depth.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(W, x2 + margin)
    y2 = min(H, y2 + margin)

    mask = np.zeros((H, W), dtype=bool)
    roi = depth[y1:y2, x1:x2]
    valid = roi > 0
    if not valid.any():
        mask[y1:y2, x1:x2] = True
        return mask

    med = np.median(roi[valid])
    std = np.std(roi[valid])
    # Keep pixels close to median depth (foreground object)
    threshold = max(std * 1.5, 0.02)
    fg = valid & (np.abs(roi - med) < threshold)
    mask[y1:y2, x1:x2] = fg
    return mask


# ── Main estimator ────────────────────────────────────────────────────────────

class PoseEstimator:
    """
    Estimates 6-DoF object poses from RGB-D images.

    Uses FoundationPose when available; otherwise falls back to depth-based
    position estimation using camera intrinsics/extrinsics and the depth map.

    Parameters
    ----------
    use_foundation_pose:
        If True, attempt to use FoundationPose. Falls back automatically
        if the package is not installed.
    mesh_dir:
        Directory containing reference meshes for FoundationPose, organized
        as ``<object_label>/mesh.obj``. Only used when FoundationPose is active.
    """

    def __init__(
        self,
        use_foundation_pose: bool = True,
        mesh_dir: Optional[str] = None,
    ):
        self.use_fp = use_foundation_pose and _FP_AVAILABLE
        self.mesh_dir = mesh_dir
        self._fp_scorer = None
        self._fp_refiner = None

        if use_foundation_pose and not _FP_AVAILABLE:
            logger.warning(
                "FoundationPose not installed — using depth-based fallback "
                "(position only, no orientation). Install from: "
                "https://github.com/NVlabs/FoundationPose"
            )

        if self.use_fp:
            self._init_foundation_pose()

    @property
    def mode(self) -> str:
        return "foundation_pose" if self.use_fp else "depth_fallback"

    def _init_foundation_pose(self):
        """Initialize FoundationPose scorer and refiner networks."""
        try:
            scorer = _fp_estimater.FoundationPose(
                model_pts=None,
                model_normals=None,
                symmetry_tfs=None,
            )
            self._fp_scorer = scorer
            logger.info("FoundationPose initialized successfully.")
        except Exception as e:
            logger.warning("FoundationPose init failed: %s — using fallback.", e)
            self.use_fp = False

    def estimate_poses(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_params: dict,
        detections: list[dict],
    ) -> list[ObjectPose]:
        """
        Estimate 6-DoF pose for each detected object.

        Parameters
        ----------
        rgb:
            RGB image as uint8 ndarray [H, W, 3].
        depth:
            Depth map as float32 ndarray [H, W] in metres.
        camera_params:
            Dict from ``MuJoCoCapture.get_camera_params()`` containing:
            ``fovy_rad``, ``f``, ``cx``, ``cy``, ``position``, ``rotation_matrix``.
        detections:
            List of detection dicts with ``"label"`` and ``"bbox_pixels"``
            (from ``DetectedObject.to_dict()``).

        Returns
        -------
        List of ``ObjectPose`` for each detection.
        """
        if self.use_fp:
            return self._estimate_foundation_pose(rgb, depth, camera_params, detections)
        return self._estimate_depth_fallback(rgb, depth, camera_params, detections)

    def _estimate_foundation_pose(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_params: dict,
        detections: list[dict],
    ) -> list[ObjectPose]:
        """
        Full 6-DoF estimation via FoundationPose.

        For each detection, creates a foreground mask from the bounding box
        and depth, then feeds it to FoundationPose for pose refinement.
        """
        f = camera_params["f"]
        cx = camera_params["cx"]
        cy = camera_params["cy"]
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ], dtype=np.float64)

        results = []
        for det in detections:
            bbox = det.get("bbox_pixels")
            label = det.get("label", "unknown")
            if bbox is None:
                results.append(self._null_pose(label))
                continue

            mask = _bbox_mask(depth, bbox)

            try:
                pose_4x4 = self._fp_scorer.register(
                    K=K,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=mask.astype(np.uint8) * 255,
                )
                # Extract position and orientation from 4x4 transform
                pos_cam = pose_4x4[:3, 3]
                rot_cam = pose_4x4[:3, :3]

                # Transform to world frame
                R_world = camera_params["rotation_matrix"]
                cam_pos = camera_params["position"]
                pos_world = cam_pos + R_world @ pos_cam

                # Convert rotation to quaternion
                rot_world = R_world @ rot_cam
                quat = self._rotmat_to_quat(rot_world)
                euler = _quat_to_euler(quat)

                results.append(ObjectPose(
                    label=label,
                    position=pos_world.tolist(),
                    quaternion=quat.tolist(),
                    euler_deg=euler.tolist(),
                    confidence=0.9,
                    pose_mode="full",
                    bbox_pixels=bbox,
                ))
            except Exception as e:
                logger.warning("FoundationPose failed for '%s': %s — using depth fallback.", label, e)
                pose = self._depth_position(depth, bbox, camera_params, label)
                results.append(pose)

        return results

    def _estimate_depth_fallback(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_params: dict,
        detections: list[dict],
    ) -> list[ObjectPose]:
        """
        Position-only estimation from depth map + camera unprojection.

        Uses median depth within a foreground-segmented bounding box region
        for robust position estimation.
        """
        results = []
        for det in detections:
            bbox = det.get("bbox_pixels")
            label = det.get("label", "unknown")
            if bbox is None:
                results.append(self._null_pose(label))
                continue
            pose = self._depth_position(depth, bbox, camera_params, label)
            results.append(pose)
        return results

    def _depth_position(
        self,
        depth: np.ndarray,
        bbox: list[float],
        camera_params: dict,
        label: str,
    ) -> ObjectPose:
        """Estimate world position from depth map using camera unprojection."""
        H, W = depth.shape[:2]
        f = camera_params["f"]
        cx = camera_params["cx"]
        cy = camera_params["cy"]
        cam_pos = camera_params["position"]
        R = camera_params["rotation_matrix"]

        # Create foreground mask and use median depth for robustness
        mask = _bbox_mask(depth, bbox)
        masked_depth = depth[mask]
        valid = masked_depth[masked_depth > 0]

        if len(valid) == 0:
            return self._null_pose(label, bbox=bbox)

        med_depth = float(np.median(valid))

        # Use bbox centre for ray direction
        x1, y1, x2, y2 = bbox
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0

        # Unproject to camera frame
        x_cam = (u - cx) / f
        y_cam = -(v - cy) / f   # flip y: image v↓ → camera y↑
        dir_cam = np.array([x_cam, y_cam, -1.0])
        point_cam = dir_cam * med_depth / np.linalg.norm(dir_cam)

        # Transform to world frame
        pos_world = cam_pos + R @ point_cam

        return ObjectPose(
            label=label,
            position=pos_world.tolist(),
            quaternion=[1.0, 0.0, 0.0, 0.0],  # identity — unknown orientation
            euler_deg=[0.0, 0.0, 0.0],
            confidence=0.7,
            pose_mode="position_only",
            bbox_pixels=bbox,
        )

    @staticmethod
    def _null_pose(label: str, bbox: Optional[list[float]] = None) -> ObjectPose:
        return ObjectPose(
            label=label,
            position=[0.0, 0.0, 0.0],
            quaternion=[1.0, 0.0, 0.0, 0.0],
            euler_deg=[0.0, 0.0, 0.0],
            confidence=0.0,
            pose_mode="position_only",
            bbox_pixels=bbox,
        )

    @staticmethod
    def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z])
