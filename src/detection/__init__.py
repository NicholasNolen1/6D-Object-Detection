"""
Detection and 3D point classification subpackage.
"""

from src.detection.detector import DetectionResult, YOLODetector, DEFAULT_BACKGROUND_CLASSES
from src.detection.classifier import (
    SegmentedObject,
    classify_point_cloud_from_masks,
    render_mask_overlays,
    INSTANCE_COLORS
)

__all__ = [
    "DetectionResult",
    "YOLODetector",
    "DEFAULT_BACKGROUND_CLASSES",
    "SegmentedObject",
    "classify_point_cloud_from_masks",
    "render_mask_overlays",
    "INSTANCE_COLORS",
]
