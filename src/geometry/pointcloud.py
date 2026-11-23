"""
Point cloud generation, filtering, and voxel consolidation from RGB-D data.
"""

from typing import Tuple, Optional
import numpy as np
import open3d as o3d
from src.geometry.camera import CameraIntrinsics


def rgbd_to_scene_point_cloud(
    depth: np.ndarray,
    color: Optional[np.ndarray],
    intrinsics: CameraIntrinsics,
    depth_scale: float = 1000.0,
    z_min: float = 0.1,
    z_max: float = 3.0
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Stage 1: Converts the ENTIRE RGB-D sensor frame into a 3D metric scene point cloud.

    Math:
      Z = Depth(u, v) / depth_scale
      X = (u - cx) * Z / fx
      Y = (v - cy) * Z / fy

    Parameters:
      depth: (H, W) uint16 (millimeters) or float32 (meters) array
      color: (H, W, 3) uint8 or float32 RGB array, or None
      intrinsics: CameraIntrinsics instance
      depth_scale: Scale factor to convert depth map to meters (1000.0 for mm)
      z_min, z_max: Metric depth cutoff boundaries in meters

    Returns:
      pts_3d: (N, 3) float32 array of valid 3D points in camera coordinates
      colors_3d: (N, 3) float32 array of RGB colors in [0, 1], or None
      valid_mask: (H, W) bool array indicating which pixels produced valid 3D points
    """
    h, w = depth.shape[:2]
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy

    # Convert depth to meters
    z = depth.astype(np.float32) / depth_scale
    valid_mask = (z > z_min) & (z < z_max) & np.isfinite(z)

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u_v = u[valid_mask]
    v_v = v[valid_mask]
    z_v = z[valid_mask]

    x_v = (u_v - cx) * z_v / fx
    y_v = (v_v - cy) * z_v / fy

    pts_3d = np.stack([x_v, y_v, z_v], axis=-1)

    colors_3d = None
    if color is not None:
        raw_colors = color[valid_mask]
        if raw_colors.max() > 1.0:
            colors_3d = raw_colors.astype(np.float32) / 255.0
        else:
            colors_3d = raw_colors.astype(np.float32)

    return pts_3d, colors_3d, valid_mask


def points_to_open3d(
    pts_3d: np.ndarray,
    colors_3d: Optional[np.ndarray] = None
) -> o3d.geometry.PointCloud:
    """Converts numpy (N, 3) point and color arrays into an Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    if len(pts_3d) > 0:
        pcd.points = o3d.utility.Vector3dVector(pts_3d.astype(np.float64))
        if colors_3d is not None and len(colors_3d) == len(pts_3d):
            pcd.colors = o3d.utility.Vector3dVector(colors_3d[:, :3].astype(np.float64))
    return pcd


def filter_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 25,
    std_ratio: float = 1.5
) -> o3d.geometry.PointCloud:
    """
    Removes depth noise and edge-bleeding 'flying pixels' using
    Statistical Outlier Removal (SOR).
    """
    if len(pcd.points) < nb_neighbors:
        return pcd
    filtered_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return filtered_pcd


def downsample_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.005
) -> o3d.geometry.PointCloud:
    """Consolidates overlapping points using a uniform 3D voxel grid."""
    if len(pcd.points) == 0:
        return pcd
    return pcd.voxel_down_sample(voxel_size=voxel_size)
