"""
Tracking and SE(3) motion estimation subpackage.
"""

from src.tracking.association import box_iou, match_detections_to_tracks
from src.tracking.registration import register_point_clouds_icp, estimate_surface_normals
from src.tracking.tracker import (
    TrackedObject,
    MultiObjectTracker,
    compute_rotation_angle_deg,
    compute_adds_error
)

__all__ = [
    "box_iou",
    "match_detections_to_tracks",
    "register_point_clouds_icp",
    "estimate_surface_normals",
    "TrackedObject",
    "MultiObjectTracker",
    "compute_rotation_angle_deg",
    "compute_adds_error",
]
