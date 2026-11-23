"""
Geometry subpackage for camera models, 3D point cloud lifting, and 3D-to-2D projection.
"""

from src.geometry.camera import CameraIntrinsics
from src.geometry.pointcloud import (
    rgbd_to_scene_point_cloud,
    points_to_open3d,
    filter_outliers,
    downsample_point_cloud
)
from src.geometry.projection import project_point_cloud_to_2d

__all__ = [
    "CameraIntrinsics",
    "rgbd_to_scene_point_cloud",
    "points_to_open3d",
    "filter_outliers",
    "downsample_point_cloud",
    "project_point_cloud_to_2d",
]
