"""
Point-Cloud-First 3D Object Detection, 6D Tracking, and Reconstruction Package.

Core Pipeline Stages:
  1. RGB-D Ingestion & 3D Scene Lifting (src.geometry)
  2. 3D-to-2D Perspective Projection (src.geometry)
  3. 2D Instance Detection on Canvas (src.detection)
  4. 3D Point Classification from 2D Masks (src.detection)
  5. Interframe SE(3) ICP Tracking & Fusion (src.tracking)
  6. Quantitative CAD Pose & Shape Benchmarking (src.evaluation)
"""

__version__ = "0.1.0"

from src.geometry.camera import CameraIntrinsics
from src.geometry.pointcloud import (
    rgbd_to_scene_point_cloud,
    points_to_open3d,
    filter_outliers,
    downsample_point_cloud,
)
from src.geometry.projection import project_point_cloud_to_2d
from src.detection.detector import YOLODetector, DetectionResult
from src.detection.classifier import (
    SegmentedObject,
    classify_point_cloud_from_masks,
    INSTANCE_COLORS,
)
from src.tracking.association import box_iou, match_detections_to_tracks
from src.tracking.registration import register_point_clouds_icp, estimate_surface_normals
from src.tracking.tracker import TrackedObject, MultiObjectTracker
from src.evaluation.metrics import (
    compute_translation_error,
    compute_rotation_error,
    compute_adds,
    compute_chamfer_distance,
    compute_grasp_success_rate,
)
from src.visualization.overlay import render_mask_overlays
from src.visualization.scene import visualize_geometries, save_point_cloud, create_composite_scene
from src.datasets.ycb import YCBMultiTrackSequence, FrameData
from src.datasets.redwood import RedwoodSampleSequence

__all__ = [
    "CameraIntrinsics",
    "rgbd_to_scene_point_cloud",
    "points_to_open3d",
    "filter_outliers",
    "downsample_point_cloud",
    "project_point_cloud_to_2d",
    "YOLODetector",
    "DetectionResult",
    "SegmentedObject",
    "classify_point_cloud_from_masks",
    "INSTANCE_COLORS",
    "box_iou",
    "match_detections_to_tracks",
    "register_point_clouds_icp",
    "estimate_surface_normals",
    "TrackedObject",
    "MultiObjectTracker",
    "compute_translation_error",
    "compute_rotation_error",
    "compute_adds",
    "compute_chamfer_distance",
    "compute_grasp_success_rate",
    "render_mask_overlays",
    "visualize_geometries",
    "save_point_cloud",
    "create_composite_scene",
    "YCBMultiTrackSequence",
    "FrameData",
    "RedwoodSampleSequence",
]
