"""
End-to-end integration test verifying genuine calculations across the entire modular pipeline.
"""

import os
import numpy as np
import pytest

from src.geometry.pointcloud import rgbd_to_scene_point_cloud
from src.geometry.projection import project_point_cloud_to_2d
from src.detection.detector import DetectionResult
from src.detection.classifier import classify_point_cloud_from_masks
from src.tracking.tracker import MultiObjectTracker
from src.datasets.ycb import YCBMultiTrackSequence


def test_pipeline_real_data_integration():
    seq_path = "data/YCBMultiTrack-Real/006_mustard_bottle_010_potted_meat_can_easy"
    if not os.path.isdir(seq_path):
        pytest.skip(f"Sequence {seq_path} not available.")

    seq = YCBMultiTrackSequence(seq_path)
    tracker = MultiObjectTracker(iou_thresh=0.15)

    # Run on 3 consecutive frames with GT masks to test end-to-end geometry and tracking
    for idx in range(3):
        frame = seq.get_frame(idx)

        # Stage 1: Full Scene Lifting
        pts_3d, colors_3d, _ = rgbd_to_scene_point_cloud(
            frame.depth, frame.color_rgb, frame.intrinsics
        )
        assert len(pts_3d) > 10000

        # Stage 2: 3D to 2D Projection
        proj_img, pt2px, valid_proj = project_point_cloud_to_2d(
            pts_3d, colors_3d, frame.intrinsics
        )
        assert np.any(valid_proj)

        # Stage 3: Detections (from ground truth masks)
        detections = []
        for obj_name, mask in frame.gt_masks.items():
            if mask.sum() > 50:
                ys, xs = np.where(mask > 0)
                detections.append(DetectionResult(
                    mask=mask,
                    box=[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                    class_name=obj_name,
                    class_id=0,
                    conf=1.0
                ))
        assert len(detections) >= 1

        # Stage 4: 3D Point Classification
        segmented, _ = classify_point_cloud_from_masks(
            pts_3d, colors_3d, pt2px, valid_proj, detections, min_points=50
        )
        for s in segmented:
            s.gt_name = s.class_name

        assert len(segmented) >= 1
        for s in segmented:
            assert s.num_points > 50

        # Stage 5: Multi-Object Tracking & SE(3) Fusion
        updates = tracker.update(segmented, gt_poses_dict=frame.gt_poses)
        assert len(updates) >= 1

    # Stage 6: Benchmark & Canonical model verification
    tracker.consolidate(voxel_size=0.003)
    assert len(tracker.tracks) >= 1

    for tid, track in tracker.tracks.items():
        assert len(track.canonical_model.points) > 100
        assert track.frames_seen == 3
        # In frame 1 and 2, interframe registration errors should have been recorded
        assert len(track.trans_errors) >= 1
        assert len(track.rot_errors) >= 1
        # Mean translation error should be within realistic real-world bounds (< 15 mm)
        assert np.mean(track.trans_errors) < 15.0
