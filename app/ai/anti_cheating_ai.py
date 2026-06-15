import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any

from mediapipe.python.solutions import face_mesh as mp_face_mesh

from app.ai.face_detector import FaceDetector
from app.ai.head_pose import HeadPoseEstimator
from app.ai.eye_gaze import EyeGazeDetector
from app.ai.phone_detector import PhoneDetector
from app.config.settings import settings

logger = logging.getLogger(__name__)


class AntiCheatingAI:
    def __init__(self):
        logger.info("Loading AI sub-modules...")
        self.face_detector  = FaceDetector()
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.head_pose      = HeadPoseEstimator()
        self.eye_gaze       = EyeGazeDetector()
        self.phone_detector = PhoneDetector()
        logger.info("All AI sub-modules loaded.")

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Face count
        face_count, _ = self.face_detector.detect(frame)
        if face_count > 1:
            return self._result(True, "multiple_faces", face_count=face_count)
        if face_count == 0:
            self.eye_gaze.reset_buffer()
            return self._result(True, "no_face_detected", face_count=0)

        # 2. Face mesh (once)
        mesh_results = self.face_mesh.process(rgb_frame)
        landmarks = (
            mesh_results.multi_face_landmarks[0].landmark
            if mesh_results.multi_face_landmarks else None
        )

        # 3. Head pose
        yaw, pitch, head_direction = self.head_pose.estimate_from_landmarks(landmarks, w, h)
        if head_direction and head_direction != "center":
            return self._result(True, head_direction, face_count, yaw, pitch, "center")

        # 4. Eye gaze (only if head also slightly off)
        gaze_direction, _, _ = self.eye_gaze.detect_from_landmarks(landmarks, w, h)
        if gaze_direction and gaze_direction != "center":
            abs_yaw   = abs(yaw)   if yaw   is not None else 0.0
            abs_pitch = abs(pitch) if pitch is not None else 0.0
            if abs_yaw >= settings.head_assist_yaw or abs_pitch >= settings.head_assist_pitch:
                return self._result(True, gaze_direction, face_count, yaw, pitch, gaze_direction)

        # 5. Phone
        phone_found, _ = self.phone_detector.detect(frame)
        if phone_found:
            return self._result(True, "phone_detected", face_count, yaw, pitch,
                                gaze_direction or "center", phone=True)

        return self._result(False, None, face_count, yaw, pitch,
                            gaze_direction or "center", phone=False)

    @staticmethod
    def _result(
        cheating: bool,
        reason: Optional[str],
        face_count: int = 0,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        gaze: Optional[str] = None,
        phone: bool = False,
    ) -> Dict[str, Any]:
        return {
            "cheating": cheating,
            "reason":   reason,
            "details": {
                "face_count":     face_count,
                "head_pose":      {"yaw": yaw, "pitch": pitch} if yaw is not None else None,
                "gaze_direction": gaze,
                "phone_detected": phone,
            },
        }

    def close(self):
        self.face_detector.close()
        self.face_mesh.close()
        self.eye_gaze.close()
