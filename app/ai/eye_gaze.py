import numpy as np
from collections import deque
from typing import Optional, Tuple


class EyeGazeDetector:
    def __init__(self):
        self.buffer = deque(maxlen=5)

    def reset_buffer(self):
        self.buffer.clear()

    def close(self):
        pass

    def _avg_point(self, landmarks, indices, w, h):
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
        return np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])

    def detect_from_landmarks(
        self, landmarks, w: int, h: int
    ) -> Tuple[str, Optional[float], Optional[float]]:
        if landmarks is None:
            return "center", None, None

        left_iris  = [468, 469, 470, 471]
        right_iris = [473, 474, 475, 476]

        l_outer = landmarks[33]
        l_inner = landmarks[133]
        r_outer = landmarks[362]
        r_inner = landmarks[263]

        l_iris_x, _ = self._avg_point(landmarks, left_iris, w, h)
        r_iris_x, _ = self._avg_point(landmarks, right_iris, w, h)

        left_ratio  = (l_iris_x - l_outer.x * w) / ((l_inner.x - l_outer.x) * w + 1e-6)
        right_ratio = (r_iris_x - r_inner.x * w) / ((r_outer.x - r_inner.x) * w + 1e-6)
        h_ratio = (left_ratio + right_ratio) / 2

        raw = "left" if h_ratio < 0.35 else ("right" if h_ratio > 0.65 else "center")
        self.buffer.append(raw)

        if len(self.buffer) < 5:
            return "center", h_ratio, None

        counts = {d: self.buffer.count(d) for d in ["left", "right", "center"]}
        return max(counts, key=counts.get), h_ratio, None
