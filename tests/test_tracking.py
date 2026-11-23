"""
Unit tests for src.tracking (association, registration, tracker).
"""

import copy
import numpy as np
import open3d as o3d
import pytest

from src.tracking.association import box_iou, match_detections_to_tracks
from src.tracking.registration import register_point_clouds_icp, estimate_surface_normals
from src.tracking.tracker import MultiObjectTracker, TrackedObject
from src.detection.classifier import SegmentedObject


def test_box_iou():
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [0.0, 0.0, 10.0, 10.0]
    box3 = [20.0, 20.0, 30.0, 30.0]
    box4 = [5.0, 0.0, 15.0, 10.0]

    assert np.isclose(box_iou(box1, box2), 1.0)
    assert np.isclose(box_iou(box1, box3), 0.0)
    # box1 and box4 overlap in [5..10, 0..10], intersection area = 50, union area = 150 -> IoU = 1/3
    assert np.isclose(box_iou(box1, box4), 1.0 / 3.0)


def test_match_detections_to_tracks():
    det_boxes = [
        [0.0, 0.0, 10.0, 10.0],    # matches track 0
        [100.0, 100.0, 120.0, 120.0],  # matches track 1
        [500.0, 500.0, 550.0, 550.0],  # unmatched detection
    ]
    track_boxes = {
        0: [1.0, 1.0, 11.0, 11.0],     # close to det 0
        1: [98.0, 99.0, 118.0, 119.0],  # close to det 1
        2: [300.0, 300.0, 320.0, 320.0], # unmatched track
    }

    matches, unmatched_dets, unmatched_tracks = match_detections_to_tracks(
        det_boxes, track_boxes, iou_thresh=0.2
    )

    assert (0, 0) in matches
    assert (1, 1) in matches
    assert 2 in unmatched_dets
    assert 2 in unmatched_tracks


def test_icp_registration():
    # Create a synthetic point cloud on a sphere
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    pcd_target = mesh.sample_points_uniformly(number_of_points=1000)

    # Apply a known small ground truth translation (5 mm in X, 3 mm in Y)
    T_gt = np.eye(4)
    T_gt[0, 3] = 0.005
    T_gt[1, 3] = 0.003

    pcd_source = copy.deepcopy(pcd_target).transform(T_gt)

    T_est, fitness, rmse = register_point_clouds_icp(
        pcd_source, pcd_target, init_transform=np.eye(4), max_distance=0.03
    )

    assert fitness > 0.95
    # The estimated transformation should align source to target, i.e. approx inv(T_gt)
    diff = np.linalg.norm((T_est @ T_gt) - np.eye(4))
    assert diff < 0.05


def test_multi_object_tracker():
    tracker = MultiObjectTracker(iou_thresh=0.2)

    # Frame 1: 2 objects
    mesh1 = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1)
    pcd1 = mesh1.sample_points_uniformly(500)
    inst1 = SegmentedObject("bottle", 39, 0.9, [10, 10, 50, 50], np.ones((50, 50)), pcd1, 500, np.arange(500))

    mesh2 = o3d.geometry.TriangleMesh.create_sphere(0.05)
    pcd2 = mesh2.sample_points_uniformly(500)
    inst2 = SegmentedObject("can", 40, 0.85, [100, 100, 150, 150], np.ones((50, 50)), pcd2, 500, np.arange(500))

    updates1 = tracker.update([inst1, inst2])
    assert len(tracker.tracks) == 2
    assert tracker.tracks[0].class_name == "bottle"
    assert tracker.tracks[1].class_name == "can"

    # Frame 2: Objects moved slightly
    inst1_moved = SegmentedObject("bottle", 39, 0.9, [12, 12, 52, 52], np.ones((50, 50)), pcd1, 500, np.arange(500))
    inst2_moved = SegmentedObject("can", 40, 0.85, [102, 102, 152, 152], np.ones((50, 50)), pcd2, 500, np.arange(500))

    updates2 = tracker.update([inst1_moved, inst2_moved])
    assert len(tracker.tracks) == 2
    assert tracker.tracks[0].frames_seen == 2
    assert tracker.tracks[1].frames_seen == 2
