"""
Bipartite spatial association between 2D detections and active 3D tracks.
"""

from typing import List, Tuple, Dict, Set
import numpy as np


def box_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Computes 2D Intersection-over-Union (IoU) between two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def match_detections_to_tracks(
    detection_boxes: List[List[float]],
    track_boxes: Dict[int, List[float]],
    iou_thresh: float = 0.15
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Associates new detections with active tracks using greedy maximum-IoU bipartite matching.

    Parameters:
      detection_boxes: List of [x1, y1, x2, y2] for each new detection (index 0..M-1)
      track_boxes: Dict of {track_id: [x1, y1, x2, y2]} for existing tracks

    Returns:
      matches: List of (det_idx, track_id) pairs
      unmatched_detections: List of det_idx with no matching track
      unmatched_tracks: List of track_id with no matching detection
    """
    track_ids = list(track_boxes.keys())
    if len(detection_boxes) == 0:
        return [], [], track_ids
    if len(track_ids) == 0:
        return [], list(range(len(detection_boxes))), []

    # Compute cost matrix (IoU)
    iou_matrix = np.zeros((len(detection_boxes), len(track_ids)), dtype=float)
    for d_idx, d_box in enumerate(detection_boxes):
        for t_idx, t_id in enumerate(track_ids):
            iou_matrix[d_idx, t_idx] = box_iou(d_box, track_boxes[t_id])

    matches: List[Tuple[int, int]] = []
    matched_dets: Set[int] = set()
    matched_tracks: Set[int] = set()

    # Greedy matching from highest IoU downwards
    while True:
        max_iou = np.max(iou_matrix)
        if max_iou < iou_thresh:
            break
        d_idx, t_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        t_id = track_ids[t_idx]

        matches.append((int(d_idx), t_id))
        matched_dets.add(int(d_idx))
        matched_tracks.add(t_id)

        # Invalidate row and column
        iou_matrix[d_idx, :] = -1.0
        iou_matrix[:, t_idx] = -1.0

    unmatched_detections = [i for i in range(len(detection_boxes)) if i not in matched_dets]
    unmatched_tracks = [t_id for t_id in track_ids if t_id not in matched_tracks]

    return matches, unmatched_detections, unmatched_tracks
