from typing import Optional, Tuple
import cv2
import numpy as np
from app.config.settings import settings

MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),
    (0.0,   -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0,  -150.0, -125.0),
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 263, 33, 287, 57]


class HeadPoseEstimator:
    def estimate_from_landmarks(
        self, landmarks, w: int, h: int
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        if not landmarks:
            return None, None, None

        image_points = np.array([
            (landmarks[idx].x * w, landmarks[idx].y * h)
            for idx in LANDMARK_IDS
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, _ = cv2.solvePnP(
            MODEL_POINTS_3D, image_points,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None, None, None

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat([rotation_mat, np.zeros((3, 1))])
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)

        pitch = float(euler[0][0])
        yaw   = float(euler[1][0])

        direction = "center"
        if abs(yaw) > settings.yaw_threshold:
            direction = "looking_left_head" if yaw < 0 else "looking_right_head"
        elif abs(pitch) > settings.pitch_threshold:
            direction = "looking_up_head" if pitch < 0 else "looking_down_head"

        return round(yaw, 2), round(pitch, 2), direction

    def close(self):
        pass
