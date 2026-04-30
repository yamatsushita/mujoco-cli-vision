"""
mujoco-cli-vision · Vision module
==================================
Scene perception layer that turns raw camera images into structured
object descriptions consumable by the Copilot CLI.

Primary components:
  analyzer  – Florence-2 based scene analyzer
  capture   – MuJoCo scene renderer + 3-D localization helpers
  server    – FastAPI REST server exposing the vision pipeline
"""

from .analyzer import SceneAnalyzer
from .capture import MuJoCoCapture
from .pose_estimator import PoseEstimator, ObjectPose

__all__ = ["SceneAnalyzer", "MuJoCoCapture", "PoseEstimator", "ObjectPose"]
