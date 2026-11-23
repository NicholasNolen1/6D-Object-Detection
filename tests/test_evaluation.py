"""
Unit tests for src.evaluation (metrics: translation, rotation, ADD-S, Chamfer distance, grasp success rate).
"""

import numpy as np
import open3d as o3d
import pytest

from src.evaluation.metrics import (
    compute_translation_error,
    compute_rotation_error,
    compute_adds,
    compute_chamfer_distance,
    compute_grasp_success_rate,
)


def test_translation_error():
    T_gt = np.eye(4)
    T_pred = np.eye(4)
    T_pred[0, 3] = 0.010  # 10 mm in X
    T_pred[1, 3] = 0.020  # 20 mm in Y

    err = compute_translation_error(T_pred, T_gt)
    expected = np.sqrt(10.0**2 + 20.0**2)
    assert np.isclose(err, expected, atol=1e-3)


def test_rotation_error():
    R_gt = np.eye(3)
    # 90 degrees around Z axis
    R_pred = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ])

    err = compute_rotation_error(R_pred, R_gt)
    assert np.isclose(err, 90.0, atol=1e-3)

    err_zero = compute_rotation_error(R_gt, R_gt)
    assert np.isclose(err_zero, 0.0, atol=1e-3)


def test_compute_adds():
    # Symmetric object: 4 points on circle
    model_pts = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0]
    ])

    T_gt = np.eye(4)
    # Pred shifted by 0.005 m (5 mm) in Z
    T_pred = np.eye(4)
    T_pred[2, 3] = 0.005

    adds = compute_adds(model_pts, T_pred, T_gt)
    assert np.isclose(adds, 5.0, atol=1e-3)


def test_chamfer_distance():
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    pcd1 = mesh.sample_points_uniformly(2000)

    # Identical point clouds should have 0 distance
    stats_self = compute_chamfer_distance(pcd1, pcd1)
    assert np.isclose(stats_self["mean_chamfer_mm"], 0.0, atol=1e-4)
    assert stats_self["f_score_5mm"] == 100.0

    # Distance to independently sampled points on the same mesh
    stats_mesh = compute_chamfer_distance(pcd1, mesh, num_eval_points=5000)
    assert stats_mesh["mean_chamfer_mm"] < 10.0
    assert stats_mesh["f_score_10mm"] > 90.0


def test_compute_grasp_success_rate():
    errors = [5.0, 10.0, 15.0, 25.0, 30.0]
    # 3 out of 5 are < 20 mm -> 60.0%
    rate = compute_grasp_success_rate(errors, threshold_mm=20.0)
    assert np.isclose(rate, 60.0)
