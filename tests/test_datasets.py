"""
Unit tests for src.datasets (RedwoodSampleSequence, YCBMultiTrackSequence).
"""

import os
import numpy as np
import pytest

from src.datasets.redwood import RedwoodSampleSequence
from src.datasets.ycb import YCBMultiTrackSequence


def test_redwood_sample_dataset():
    dataset = RedwoodSampleSequence(max_frames=3)
    assert len(dataset) == 3

    color, depth, intrinsics = dataset.get_frame(0)
    assert color.ndim == 3 and color.shape[2] == 3
    assert depth.ndim == 2
    assert intrinsics.width == 640
    assert intrinsics.height == 480


def test_ycb_dataset_loader():
    seq_path = "data/YCBMultiTrack-Real/006_mustard_bottle_010_potted_meat_can_easy"
    if not os.path.isdir(seq_path):
        pytest.skip(f"YCB sequence '{seq_path}' not found on disk.")

    seq = YCBMultiTrackSequence(seq_path)
    assert len(seq) > 0
    assert "006_mustard_bottle" in seq.object_names

    frame = seq.get_frame(0)
    assert frame.color_rgb.ndim == 3
    assert frame.depth.ndim == 2
    assert frame.intrinsics.fx > 0
    assert "006_mustard_bottle" in frame.gt_poses
    assert frame.gt_poses["006_mustard_bottle"].shape == (4, 4)
