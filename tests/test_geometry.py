"""
Unit tests for src.geometry (CameraIntrinsics, rgbd_to_scene_point_cloud, project_point_cloud_to_2d).
"""

import tempfile
import numpy as np
import open3d as o3d
import pytest

from src.geometry.camera import CameraIntrinsics
from src.geometry.pointcloud import (
    rgbd_to_scene_point_cloud,
    points_to_open3d,
    filter_outliers,
    downsample_point_cloud,
)
from src.geometry.projection import project_point_cloud_to_2d


def test_camera_intrinsics_basics():
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    intrinsics = CameraIntrinsics.from_matrix(K, width=640, height=480)

    assert intrinsics.fx == 500.0
    assert intrinsics.fy == 500.0
    assert intrinsics.cx == 320.0
    assert intrinsics.cy == 240.0
    assert intrinsics.width == 640
    assert intrinsics.height == 480
    assert np.allclose(intrinsics.matrix, K)


def test_camera_intrinsics_file_io():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as tmp:
        tmp.write("600.0 0.0 300.0\n0.0 600.0 200.0\n0.0 0.0 1.0\n")
        tmp_path = tmp.name

    intrinsics = CameraIntrinsics.from_file(tmp_path, width=600, height=400)
    assert intrinsics.fx == 600.0
    assert intrinsics.fy == 600.0
    assert intrinsics.cx == 300.0
    assert intrinsics.cy == 200.0


def test_rgbd_to_scene_point_cloud_math():
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    depth = np.zeros((480, 640), dtype=np.uint16)
    color = np.zeros((480, 640, 3), dtype=np.uint8)

    # Place a single point at pixel (u=320, v=240) with depth 1.0 meter (1000 mm)
    depth[240, 320] = 1000
    color[240, 320] = [255, 128, 0]

    # Place another point at pixel (u=420, v=290) with depth 2.0 meters (2000 mm)
    # X = (420 - 320) * 2.0 / 500.0 = 100 * 2.0 / 500 = 0.4 meters
    # Y = (290 - 240) * 2.0 / 500.0 = 50 * 2.0 / 500 = 0.2 meters
    depth[290, 420] = 2000
    color[290, 420] = [0, 255, 100]

    pts_3d, colors_3d, valid_mask = rgbd_to_scene_point_cloud(depth, color, intrinsics)

    assert len(pts_3d) == 2
    assert colors_3d is not None and len(colors_3d) == 2

    # Verify first point is at origin (0, 0, 1.0)
    pt0 = pts_3d[0]
    assert np.isclose(pt0[0], 0.0, atol=1e-4)
    assert np.isclose(pt0[1], 0.0, atol=1e-4)
    assert np.isclose(pt0[2], 1.0, atol=1e-4)
    assert np.allclose(colors_3d[0], [1.0, 128 / 255.0, 0.0], atol=1e-2)

    # Verify second point is at (0.4, 0.2, 2.0)
    pt1 = pts_3d[1]
    assert np.isclose(pt1[0], 0.4, atol=1e-4)
    assert np.isclose(pt1[1], 0.2, atol=1e-4)
    assert np.isclose(pt1[2], 2.0, atol=1e-4)


def test_project_point_cloud_to_2d():
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)

    # Points in camera frame
    pts_3d = np.array([
        [0.0, 0.0, 1.0],      # Should project to (320, 240)
        [0.4, 0.2, 2.0],      # Should project to (420, 290)
        [0.0, 0.0, -1.0],     # Invalid Z (behind camera)
        [100.0, 100.0, 1.0],  # Out of image bounds
    ], dtype=np.float32)

    colors_3d = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=np.float32)

    proj_img, pt2px, valid_proj = project_point_cloud_to_2d(pts_3d, colors_3d, intrinsics)

    assert valid_proj[0] == True
    assert valid_proj[1] == True
    assert valid_proj[2] == False
    assert valid_proj[3] == False

    assert pt2px[0, 0] == 320 and pt2px[0, 1] == 240
    assert pt2px[1, 0] == 420 and pt2px[1, 1] == 290

    # Check color at (240, 320) is red
    assert proj_img[240, 320, 0] == 255
    # Check color at (290, 420) is green
    assert proj_img[290, 420, 1] == 255


def test_pointcloud_filtering_and_downsampling():
    # Create 100 points close together + 1 distant outlier
    pts = np.random.normal(loc=[0, 0, 1], scale=0.01, size=(200, 3))
    outlier = np.array([[10.0, 10.0, 10.0]])
    all_pts = np.vstack([pts, outlier])

    pcd = points_to_open3d(all_pts)
    assert len(pcd.points) == 201

    filtered = filter_outliers(pcd, nb_neighbors=25, std_ratio=1.5)
    assert len(filtered.points) <= 200

    downsampled = downsample_point_cloud(filtered, voxel_size=0.02)
    assert len(downsampled.points) < len(filtered.points)
