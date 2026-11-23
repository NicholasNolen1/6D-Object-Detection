"""
Unit tests for src.detection (YOLODetector, DetectionResult, classify_point_cloud_from_masks).
"""

import numpy as np
import pytest

from src.detection.detector import DetectionResult, YOLODetector
from src.detection.classifier import (
    SegmentedObject,
    classify_point_cloud_from_masks,
    render_mask_overlays,
)


def test_detection_result_dataclass():
    mask = np.ones((100, 100), dtype=np.uint8)
    box = [10.0, 10.0, 50.0, 50.0]
    det = DetectionResult(mask=mask, box=box, class_name="cup", class_id=41, conf=0.88)

    assert det.class_name == "cup"
    assert det.conf == 0.88
    assert det.box == [10.0, 10.0, 50.0, 50.0]
    assert det.mask.shape == (100, 100)


def test_yolo_fallback_regions():
    regions = YOLODetector.fallback_regions((480, 640))
    assert len(regions) >= 1
    assert regions[0].class_name == "center_object"
    assert regions[0].mask.shape == (480, 640)
    assert regions[0].mask.sum() > 0


def test_classify_point_cloud_from_masks():
    # Create 300 3D points
    np.random.seed(42)
    pts_3d = np.random.uniform(-0.5, 0.5, size=(300, 3)).astype(np.float32)
    pts_3d[:, 2] += 1.0  # Depth around 1.0m
    colors_3d = np.ones((300, 3), dtype=np.float32)

    # Point to pixel map
    pt2px = np.zeros((300, 2), dtype=int)
    # First 100 points land in region [50:150, 50:150] (Object 1)
    pt2px[:100, 0] = np.random.randint(50, 150, size=100)
    pt2px[:100, 1] = np.random.randint(50, 150, size=100)

    # Second 100 points land in region [200:300, 200:300] (Object 2)
    pt2px[100:200, 0] = np.random.randint(200, 300, size=100)
    pt2px[100:200, 1] = np.random.randint(200, 300, size=100)

    # Last 100 points land in background
    pt2px[200:, 0] = np.random.randint(400, 500, size=100)
    pt2px[200:, 1] = np.random.randint(400, 500, size=100)

    valid_proj = np.ones(300, dtype=bool)

    # Define 2 detection masks
    mask1 = np.zeros((480, 640), dtype=np.uint8)
    mask1[50:150, 50:150] = 1

    mask2 = np.zeros((480, 640), dtype=np.uint8)
    mask2[200:300, 200:300] = 1

    detections = [
        DetectionResult(mask=mask1, box=[50.0, 50.0, 150.0, 150.0], class_name="bottle", class_id=39, conf=0.9),
        DetectionResult(mask=mask2, box=[200.0, 200.0, 300.0, 300.0], class_name="can", class_id=40, conf=0.85),
    ]

    segmented, labels = classify_point_cloud_from_masks(
        pts_3d, colors_3d, pt2px, valid_proj, detections, min_points=50
    )

    assert len(segmented) == 2
    assert segmented[0].class_name == "bottle"
    assert len(segmented[0].point_indices) == 100
    assert segmented[0].num_points >= 80
    assert segmented[1].class_name == "can"
    assert len(segmented[1].point_indices) == 100
    assert segmented[1].num_points >= 80

    assert np.all(labels[:100] == 1)
    assert np.all(labels[100:200] == 2)
    assert np.all(labels[200:] == 0)


def test_render_mask_overlays():
    color_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[50:100, 50:100] = 1

    obj = SegmentedObject(
        class_name="cup",
        class_id=41,
        conf=0.95,
        box=[50.0, 50.0, 100.0, 100.0],
        mask_2d=mask,
        pcd=None,
        num_points=120,
        point_indices=np.arange(120)
    )

    vis = render_mask_overlays(color_bgr, [obj])
    assert vis.shape == (480, 640, 3)
    # The overlay region must not be completely black
    assert vis[75, 75].sum() > 0
