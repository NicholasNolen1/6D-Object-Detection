"""
3D Point Cloud Classifier: Classifies individual 3D points from 2D projected masks.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import open3d as o3d
import cv2

from src.detection.detector import DetectionResult
from src.geometry.pointcloud import points_to_open3d, filter_outliers

# Distinct RGB colors for visual instance rendering
INSTANCE_COLORS = [
    [0.90, 0.20, 0.20],  # Red
    [0.20, 0.70, 0.30],  # Green
    [0.20, 0.40, 0.90],  # Blue
    [0.90, 0.75, 0.10],  # Yellow
    [0.70, 0.20, 0.85],  # Purple
    [0.10, 0.80, 0.85],  # Cyan
    [0.95, 0.50, 0.10],  # Orange
    [0.85, 0.30, 0.50],  # Pink
]


@dataclass
class SegmentedObject:
    """Represents a 3D object point cloud isolated and classified from the scene."""
    class_name: str
    class_id: int
    conf: float
    box: List[float]
    mask_2d: np.ndarray
    pcd: o3d.geometry.PointCloud
    num_points: int
    point_indices: np.ndarray
    gt_name: Optional[str] = None


def classify_point_cloud_from_masks(
    pts_3d: np.ndarray,
    colors_3d: Optional[np.ndarray],
    point_to_pixel_map: np.ndarray,
    valid_proj: np.ndarray,
    detections: List[DetectionResult],
    min_points: int = 50
) -> Tuple[List[SegmentedObject], np.ndarray]:
    """
    Stage 4: Classifies the individual points of the 3D Point Cloud using 2D projected masks.

    Math:
      Point p_i in P belongs to Object k <=> Mask_k(u_i, v_i) == 1

    Parameters:
      pts_3d: (N, 3) float32 array of 3D scene points
      colors_3d: (N, 3) float32 array of point colors, or None
      point_to_pixel_map: (N, 2) int array of projected coordinates (u, v)
      valid_proj: (N,) bool array indicating valid 2D projections
      detections: List of 2D DetectionResult instances from Stage 3
      min_points: Minimum 3D points required to retain an object instance

    Returns:
      segmented_instances: List[SegmentedObject] containing isolated 3D geometry
      point_labels: (N,) int array assigning each 3D point to an instance ID (0 = unclassified / background)
    """
    point_labels = np.zeros(len(pts_3d), dtype=int)
    segmented_instances: List[SegmentedObject] = []

    if len(pts_3d) == 0:
        return segmented_instances, point_labels

    u = point_to_pixel_map[:, 0]
    v = point_to_pixel_map[:, 1]

    for inst_idx, det in enumerate(detections):
        mask_2d = det.mask
        inst_id = inst_idx + 1
        mh, mw = mask_2d.shape[:2]

        # Evaluate 3D point membership: does the point's projection land on this mask?
        is_point_in_mask = np.zeros(len(pts_3d), dtype=bool)
        in_bounds = valid_proj & (v >= 0) & (v < mh) & (u >= 0) & (u < mw)
        is_point_in_mask[in_bounds] = mask_2d[v[in_bounds], u[in_bounds]] > 0

        # Assign instance label to point
        point_labels[is_point_in_mask] = inst_id

        obj_pts = pts_3d[is_point_in_mask]
        obj_colors = colors_3d[is_point_in_mask] if colors_3d is not None else None

        if len(obj_pts) >= min_points:
            obj_pcd = points_to_open3d(obj_pts, obj_colors)
            obj_pcd = filter_outliers(obj_pcd, nb_neighbors=25, std_ratio=1.5)

            segmented_instances.append(SegmentedObject(
                class_name=det.class_name,
                class_id=det.class_id,
                conf=det.conf,
                box=det.box,
                mask_2d=mask_2d,
                pcd=obj_pcd,
                num_points=len(obj_pcd.points),
                point_indices=np.where(is_point_in_mask)[0]
            ))

    return segmented_instances, point_labels


def render_mask_overlays(
    color_bgr: np.ndarray,
    instances: List[SegmentedObject]
) -> np.ndarray:
    """
    Creates an alpha-blended visual overlay showing all isolated 2D masks,
    bounding boxes, and 3D point counts for inspection.
    """
    vis_img = color_bgr.copy()
    overlay = np.zeros_like(color_bgr, dtype=np.uint8)

    for i, inst in enumerate(instances):
        color_rgb = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        color_bgr_inst = (int(color_rgb[2] * 255), int(color_rgb[1] * 255), int(color_rgb[0] * 255))

        mask = inst.mask_2d > 0
        overlay[mask] = color_bgr_inst

        x1, y1, x2, y2 = [int(val) for val in inst.box]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color_bgr_inst, 2)
        lbl = f"[{i}] {inst.class_name} ({inst.num_points} pts, conf={inst.conf:.2f})"
        cv2.putText(vis_img, lbl, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis_img, lbl, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr_inst, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.45, vis_img, 0.55, 0, vis_img)
    return vis_img
