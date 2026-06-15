# syntax=docker/dockerfile:1
# -------------------------------------------------------
# Anti-Cheating AI — Optimized for Railway free tier
# Python 3.11-slim, CPU-only PyTorch
# -------------------------------------------------------

FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV & MediaPipe (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Install PyTorch CPU-only FIRST (largest package, better layer caching)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/    ./app/
COPY models/ ./models/
COPY main.py .

# Pre-warm YOLO model at build time (avoids slow startup)
RUN python -c "from ultralytics import YOLO; YOLO('models/yolov8n.pt')"

RUN chown -R appuser:appgroup /app
USER appuser

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
