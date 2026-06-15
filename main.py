import time
import cv2
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.ai.anti_cheating_ai import AntiCheatingAI
from app.utils.logger import get_logger

logger = get_logger(__name__)

ai: Optional[AntiCheatingAI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ai
    logger.info("Loading AI models...")
    ai = AntiCheatingAI()
    logger.info("Ready.")
    yield
    if ai:
        ai.close()


app = FastAPI(title="Anti-Cheating AI", version="1.0.0", lifespan=lifespan)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}


class HeadPose(BaseModel):
    yaw: float
    pitch: float


class AnalyzeResponse(BaseModel):
    cheating: bool
    reason: Optional[str]
    face_count: int
    head_pose: Optional[HeadPose]
    gaze_direction: Optional[str]
    processing_time_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    if ai is None:
        raise HTTPException(status_code=503, detail="AI not loaded yet")

    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported type: {file.content_type}")

    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")

        frame = await run_in_threadpool(
            cv2.imdecode, np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Image decode error: {exc}")
        raise HTTPException(status_code=400, detail=f"Image read failed: {exc}")

    try:
        t0 = time.perf_counter()
        result = await run_in_threadpool(ai.analyze, frame)
        t_ms = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as exc:
        logger.error(f"AI inference error: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    details = result.get("details", {})
    hp = details.get("head_pose")

    return AnalyzeResponse(
        cheating=result["cheating"],
        reason=result.get("reason"),
        face_count=details.get("face_count", 0),
        head_pose=HeadPose(**hp) if hp else None,
        gaze_direction=details.get("gaze_direction"),
        processing_time_ms=t_ms,
    )
