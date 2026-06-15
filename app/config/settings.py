import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = int(os.environ.get("PORT", 8000))

    # AI thresholds
    yaw_threshold: float = 20.0
    pitch_threshold: float = 15.0
    head_assist_yaw: float = 10.0
    head_assist_pitch: float = 8.0
    phone_confidence: float = 0.40

    # Model path
    yolo_model_path: str = "models/yolov8n.pt"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
