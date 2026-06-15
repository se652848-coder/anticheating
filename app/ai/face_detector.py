from typing import List, Tuple
import cv2
import numpy as np
from mediapipe.python.solutions import face_detection as mp_face_detection


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        self.detector = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame: np.ndarray) -> Tuple[int, List]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        if not results.detections:
            return 0, []
        return len(results.detections), results.detections

    def close(self):
        self.detector.close()
