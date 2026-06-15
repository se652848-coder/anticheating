from fastapi import APIRouter, HTTPException
from app.dependencies import get_ai
from app.domain.entities import HealthResponse, RootResponse

router = APIRouter(tags=["meta"])


@router.get("/", response_model=RootResponse)
def root():
    return RootResponse(
        service="Anti-Cheating AI Microservice",
        version="1.0.0",
        status="running",
        endpoints={
            "health":  "GET  /health",
            "analyze": "POST /analyze",
            "docs":    "GET  /docs",
        },
    )


@router.get("/health", response_model=HealthResponse)
def health():
    if get_ai() is None:
        raise HTTPException(status_code=503, detail="AI models not loaded yet")
    return HealthResponse(status="ok")
