"""
Anti-Cheating AI Microservice — FastAPI entry point.
Run with: uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config.settings import settings
from app.dependencies import initialize_dependencies, get_ai
from app.presentation.routes import router as analyze_router
from app.presentation.health import router as health_router
from app.utils.exceptions import AntiCheatingError
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Anti-Cheating service")
    try:
        await initialize_dependencies()
        logger.info("Anti-Cheating service ready")
    except Exception as e:
        logger.error(f"Failed to initialize dependencies: {e}", exc_info=True)
        raise
    yield
    ai = get_ai()
    if ai:
        ai.close()
    logger.info("Anti-Cheating service shutting down")


app = FastAPI(
    title="Anti-Cheating AI Microservice",
    description=(
        "Detects cheating behavior in exam video frames. "
        "Detects multiple faces, suspicious head pose, abnormal eye gaze, and phone presence."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(AntiCheatingError)
async def anticheating_error_handler(_request: Request, exc: AntiCheatingError):
    logger.error(f"Anti-cheating error: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.message},
    )


@app.exception_handler(Exception)
async def generic_error_handler(_request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"},
    )


app.include_router(health_router)
app.include_router(analyze_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
