from pathlib import Path

import cv2
import numpy as np

from .config import CONF_THRESHOLD, IMGSZ
from .schemas import Region


class Detector:
    """Wraps the two YOLO segmentation models (car parts and damage)."""

    def __init__(self, parts_weights, damage_weights, conf=CONF_THRESHOLD):
        for path in (parts_weights, damage_weights):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Missing weights file: {path}. Train the models with train.py first (see README)."
                )
        from ultralytics import YOLO  # imported here so the rest of the app works without it

        self.parts_model = YOLO(str(parts_weights))
        self.damage_model = YOLO(str(damage_weights))
        self.conf = conf

    def _run(self, model, image):
        h, w = image.shape[:2]
        result = model.predict(image, conf=self.conf, imgsz=IMGSZ, verbose=False)[0]
        regions = []
        if result.boxes is None or len(result.boxes) == 0:
            return regions
        polys = result.masks.xy if result.masks is not None else None
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            mask = np.zeros((h, w), np.uint8)
            if polys is not None and len(polys[i]) >= 3:
                cv2.fillPoly(mask, [np.round(polys[i]).astype(np.int32)], 1)
            else:  # detection-only model: fall back to the bounding box
                mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 1
            label = result.names[int(box.cls[0])].lower().replace("_", " ")
            regions.append(Region(label, float(box.conf[0]), mask.astype(bool), (x1, y1, x2, y2)))
        return regions

    def predict(self, image):
        """Return (parts, damages) for a BGR image."""
        return self._run(self.parts_model, image), self._run(self.damage_model, image)
