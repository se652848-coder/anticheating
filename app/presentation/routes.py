import time
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.dependencies import get_ai
from app.domain.entities import AnalyzeResponse, AnalysisDetails, HeadPose
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["inference"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    """
    Analyze an image frame for cheating behavior.

    - **file**: image file (JPEG, PNG, WebP, BMP)

    Returns cheating verdict with face count, head pose, gaze direction,
    and phone detection.
    """
    ai = get_ai()
    if ai is None:
        raise HTTPException(status_code=503, detail="AI models not loaded")

    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}",
        )

    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        np_arr = np.frombuffer(raw_bytes, np.uint8)
        frame  = await run_in_threadpool(cv2.imdecode, np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Image decode error: {exc}")
        raise HTTPException(status_code=400, detail=f"Image read failed: {exc}")

    try:
        t_start = time.perf_counter()
        result  = await run_in_threadpool(ai.analyze, frame)
        t_ms    = round((time.perf_counter() - t_start) * 1000, 2)
    except Exception as exc:
        logger.error(f"AI inference error: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    details = result.get("details", {})
    hp_raw  = details.get("head_pose")
    head_pose_obj = HeadPose(**hp_raw) if hp_raw else None

    return AnalyzeResponse(
        cheating=result["cheating"],
        reason=result.get("reason"),
        details=AnalysisDetails(
            face_count=details.get("face_count", 0),
            head_pose=head_pose_obj,
            gaze_direction=details.get("gaze_direction"),
            phone_detected=details.get("phone_detected", False),
        ),
        processing_time_ms=t_ms,
    )
