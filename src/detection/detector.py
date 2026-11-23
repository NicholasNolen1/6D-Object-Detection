"""
2D Instance Segmentation Detector on Projected Canvas Planes.
"""

from dataclasses import dataclass
from typing import List, Optional, Set
import numpy as np
import cv2


@dataclass
class DetectionResult:
    """Represents a single 2D object instance detection."""
    mask: np.ndarray       # (H, W) uint8 binary mask
    box: List[float]       # [x1, y1, x2, y2]
    class_name: str
    class_id: int
    conf: float


DEFAULT_BACKGROUND_CLASSES: Set[str] = {
    "dining table", "chair", "bed", "couch", "refrigerator", "toilet", "bench"
}


class YOLODetector:
    """Wrapper around YOLOv8/11 instance segmentation models."""
    def __init__(self, model_path: str = "yolov8n-seg.pt"):
        self.model_path = model_path
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load YOLO model from '{self.model_path}': {e}")
        return self._model

    def detect(
        self,
        proj_image: np.ndarray,
        conf_thresh: float = 0.20,
        filter_background: bool = True,
        min_pixels: int = 80
    ) -> List[DetectionResult]:
        """
        Runs instance segmentation on the 2D projected image canvas.
        Isolates ALL detected object masks.
        """
        h, w = proj_image.shape[:2]
        model = self._load_model()
        results = model(proj_image, verbose=False, conf=conf_thresh)

        if len(results) == 0 or results[0].masks is None:
            return []

        r = results[0]
        masks_data = r.masks.data.cpu().numpy()
        boxes_data = r.boxes.data.cpu().numpy()

        detections: List[DetectionResult] = []
        for i, box in enumerate(boxes_data):
            cls_id = int(box[5])
            conf = float(box[4])
            cls_name = model.names.get(cls_id, f"obj_{cls_id}")

            if filter_background and cls_name in DEFAULT_BACKGROUND_CLASSES:
                continue

            raw_mask = masks_data[i]
            mask = cv2.resize(
                (raw_mask > 0.5).astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

            if mask.sum() < min_pixels:
                continue

            detections.append(DetectionResult(
                mask=mask,
                box=box[:4].tolist(),
                class_name=cls_name,
                class_id=cls_id,
                conf=conf
            ))

        return detections

    @staticmethod
    def fallback_regions(image_shape: tuple) -> List[DetectionResult]:
        """Heuristic regional fallback masks when YOLO is unavailable or yields no detections."""
        h, w = image_shape[:2]
        cy, cx = h // 2, w // 2
        rh, rw = h // 4, w // 4

        mask_center = np.zeros((h, w), dtype=np.uint8)
        mask_center[cy - rh : cy + rh, cx - rw : cx + rw] = 1

        return [
            DetectionResult(
                mask=mask_center,
                box=[float(cx - rw), float(cy - rh), float(cx + rw), float(cy + rh)],
                class_name="center_object",
                class_id=0,
                conf=1.0
            )
        ]
