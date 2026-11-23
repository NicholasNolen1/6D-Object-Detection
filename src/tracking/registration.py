"""
3D Point Cloud SE(3) registration and motion estimation using Point-to-Plane ICP.
"""

from typing import Tuple, Optional
import numpy as np
import open3d as o3d


def estimate_surface_normals(
    pcd: o3d.geometry.PointCloud,
    radius: float = 0.03,
    max_nn: int = 30
) -> None:
    """Estimates surface normals in place using hybrid KDTree search."""
    if not pcd.has_normals() and len(pcd.points) > 3:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )


def register_point_clouds_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    init_transform: Optional[np.ndarray] = None,
    max_distance: float = 0.04,
    method: str = "point_to_plane",
    max_iterations: int = 40
) -> Tuple[np.ndarray, float, float]:
    """
    Computes rigid SE(3) transformation [R | t] aligning source_pcd to target_pcd.

    Math:
      Find T in SE(3) minimizing: sum_i || (T * p_i - q_i) . n_i ||^2

    Parameters:
      source_pcd: Source point cloud at time t (to be aligned)
      target_pcd: Target point cloud at time t-1 (reference)
      init_transform: (4, 4) initial transformation guess (e.g., constant-velocity prior)
      max_distance: Maximum correspondence distance in meters
      method: 'point_to_plane' or 'point_to_point'
      max_iterations: Maximum solver iterations

    Returns:
      transformation: (4, 4) SE(3) transformation matrix mapping source -> target
      fitness: Overlap ratio of inlier correspondences [0, 1]
      inlier_rmse: Root-mean-square error of inlier correspondences in meters
    """
    if init_transform is None:
        init_transform = np.eye(4)

    estimate_surface_normals(source_pcd)
    estimate_surface_normals(target_pcd)

    if method == "point_to_plane" and source_pcd.has_normals() and target_pcd.has_normals():
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)

    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        max_distance,
        init_transform,
        estimation_method,
        criteria
    )

    return result.transformation, float(result.fitness), float(result.inlier_rmse)
