from typing import Optional, Dict
from pydantic import BaseModel


class HeadPose(BaseModel):
    yaw: float
    pitch: float


class AnalysisDetails(BaseModel):
    face_count: int
    head_pose: Optional[HeadPose] = None
    gaze_direction: Optional[str] = None
    phone_detected: bool


class AnalyzeResponse(BaseModel):
    cheating: bool
    reason: Optional[str] = None
    details: AnalysisDetails
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str


class RootResponse(BaseModel):
    service: str
    version: str
    status: str
    endpoints: Dict[str, str]
