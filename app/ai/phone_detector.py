from pathlib import Path
from typing import List, Tuple
import numpy as np
from app.config.settings import settings

PHONE_CLASS_ID = 67  # COCO: "cell phone"

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


class PhoneDetector:
    def __init__(self):
        if not _YOLO_AVAILABLE:
            raise ImportError("ultralytics is not installed.")
        model_path = settings.yolo_model_path
        self.model = YOLO(model_path if Path(model_path).exists() else "yolov8n.pt")

    def detect(self, frame: np.ndarray) -> Tuple[bool, List[dict]]:
        results = self.model(frame, verbose=False, imgsz=320)[0]
        phones = [
            {
                "confidence": round(float(box.conf[0]), 3),
                "box": {k: int(v) for k, v in zip(
                    ["x1", "y1", "x2", "y2"], box.xyxy[0].tolist()
                )},
            }
            for box in results.boxes
            if int(box.cls[0]) == PHONE_CLASS_ID
            and float(box.conf[0]) >= settings.phone_confidence
        ]
        return len(phones) > 0, phones
