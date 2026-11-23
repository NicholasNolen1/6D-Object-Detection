"""
Quantitative evaluation metrics: ADD-S, Chamfer Distance, F-Score, and Pose Errors.
"""

from typing import Dict, List, Tuple
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def compute_translation_error(T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    """Computes translation centroid error in millimeters."""
    diff = T_pred[:3, 3] - T_gt[:3, 3]
    return float(np.linalg.norm(diff) * 1000.0)


def compute_rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Computes geodesic angular rotation error in degrees."""
    R_err = R_pred @ R_gt.T
    trace = np.clip(np.trace(R_err), -1.0, 3.0)
    angle_rad = np.arccos((trace - 1.0) / 2.0)
    return float(np.degrees(angle_rad))


def compute_adds(model_points: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    """
    Computes ADD-S (Average Distance for Symmetric objects) in millimeters.
    Math:
      ADD-S = (1 / m) * sum_{p1 in M} min_{p2 in M} || (R_gt * p1 + t_gt) - (R_pred * p2 + t_pred) ||
    """
    pts_gt = (T_gt[:3, :3] @ model_points.T + T_gt[:3, 3:4]).T
    pts_pred = (T_pred[:3, :3] @ model_points.T + T_pred[:3, 3:4]).T

    tree = cKDTree(pts_gt)
    dists, _ = tree.query(pts_pred, k=1)
    return float(np.mean(dists) * 1000.0)


def compute_chamfer_distance(
    reconstructed_pcd: o3d.geometry.PointCloud,
    cad_mesh_or_pcd: o3d.geometry.Geometry3D,
    num_eval_points: int = 10000
) -> Dict[str, float]:
    """
    Computes surface distance metrics between a reconstructed point cloud and the ground-truth CAD model.

    Returns dict with:
      mean_chamfer_mm: Mean distance in mm
      median_chamfer_mm: Median distance in mm
      f_score_5mm: Percentage of points within 5 mm [0, 100]
      f_score_10mm: Percentage of points within 10 mm [0, 100]
    """
    if isinstance(cad_mesh_or_pcd, o3d.geometry.TriangleMesh):
        eval_cad_pcd = cad_mesh_or_pcd.sample_points_uniformly(number_of_points=num_eval_points)
    else:
        eval_cad_pcd = cad_mesh_or_pcd

    if len(reconstructed_pcd.points) == 0 or len(eval_cad_pcd.points) == 0:
        return {
            "mean_chamfer_mm": float("inf"),
            "median_chamfer_mm": float("inf"),
            "f_score_5mm": 0.0,
            "f_score_10mm": 0.0
        }

    dists = np.asarray(reconstructed_pcd.compute_point_cloud_distance(eval_cad_pcd))
    dists_mm = dists * 1000.0

    return {
        "mean_chamfer_mm": float(np.mean(dists_mm)),
        "median_chamfer_mm": float(np.median(dists_mm)),
        "f_score_5mm": float(np.mean(dists < 0.005) * 100.0),
        "f_score_10mm": float(np.mean(dists < 0.010) * 100.0)
    }


def compute_grasp_success_rate(adds_errors_mm: List[float], threshold_mm: float = 20.0) -> float:
    """Calculates the percentage of frames achieving ADD-S error < threshold_mm (standard: 20 mm)."""
    if not adds_errors_mm:
        return 0.0
    arr = np.array(adds_errors_mm)
    return float(np.mean(arr < threshold_mm) * 100.0)
