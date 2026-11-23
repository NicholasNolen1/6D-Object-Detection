"""
2D Visualization overlays for instance masks, bounding boxes, and object labels.
"""

from typing import List, Optional, Tuple
import numpy as np
import cv2

from src.detection.classifier import SegmentedObject, INSTANCE_COLORS


def render_mask_overlays(
    color_bgr: np.ndarray,
    instances: List[SegmentedObject],
    palette: Optional[List[List[float]]] = None,
    alpha: float = 0.45
) -> np.ndarray:
    """
    Renders semi-transparent color masks and bounding box labels onto a 2D image.

    Parameters:
      color_bgr: (H, W, 3) uint8 BGR image
      instances: List of SegmentedObject instances containing 2D masks and metadata
      palette: Optional list of RGB colors in [0, 1] range
      alpha: Transparency factor for mask overlay blending

    Returns:
      vis_img: Annotated BGR image with overlays and bounding boxes
    """
    colors = palette or INSTANCE_COLORS
    vis_img = color_bgr.copy()
    overlay = np.zeros_like(color_bgr, dtype=np.uint8)

    for i, inst in enumerate(instances):
        color_rgb = colors[i % len(colors)]
        color_bgr_inst = (int(color_rgb[2] * 255), int(color_rgb[1] * 255), int(color_rgb[0] * 255))

        mask = inst.mask_2d > 0
        overlay[mask] = color_bgr_inst

        x1, y1, x2, y2 = [int(v) for v in inst.box]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color_bgr_inst, 2)
        lbl = f"[{i}] {inst.class_name} ({inst.num_points} pts, conf={inst.conf:.2f})"
        cv2.putText(vis_img, lbl, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis_img, lbl, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr_inst, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, vis_img, 1.0 - alpha, 0, vis_img)
    return vis_img


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Draws 2D bounding boxes and optional text labels onto an image."""
    vis = image.copy()
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if labels and idx < len(labels):
            lbl = labels[idx]
            cv2.putText(vis, lbl, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis
