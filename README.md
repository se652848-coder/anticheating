# Anti-Cheating AI Microservice

FastAPI microservice that detects cheating behavior in exam video frames using computer vision.

## Detection Capabilities
- **Multiple faces** — more than one person in frame
- **No face** — student left the frame
- **Head pose** — looking left/right/up/down beyond threshold
- **Eye gaze** — eyes deviated while head also slightly turned
- **Phone detection** — mobile phone visible in frame (YOLOv8)

## Project Structure

```
anticheating/
├── main.py                         # FastAPI entry point
├── Dockerfile                      # Railway/Docker deployment
├── railway.json                    # Railway configuration
├── requirements.txt
├── .env.example
├── models/
│   └── yolov8n.pt                  # YOLOv8 nano model
└── app/
    ├── config/
    │   └── settings.py             # Pydantic settings (env vars)
    ├── domain/
    │   └── entities.py             # Pydantic request/response schemas
    ├── ai/
    │   ├── anti_cheating_ai.py     # Main AI orchestrator
    │   ├── face_detector.py        # MediaPipe face detection
    │   ├── head_pose.py            # Head pose estimation (solvePnP)
    │   ├── eye_gaze.py             # Iris landmark gaze tracking
    │   └── phone_detector.py       # YOLOv8 phone detection
    ├── presentation/
    │   ├── health.py               # GET / and GET /health
    │   └── routes.py               # POST /analyze
    ├── utils/
    │   ├── exceptions.py
    │   └── logger.py
    └── dependencies.py             # App-level AI singleton
```

## API Endpoints

| Method | Path       | Description                        |
|--------|------------|------------------------------------|
| GET    | `/`        | Service info                       |
| GET    | `/health`  | Health check (used by Railway)     |
| POST   | `/analyze` | Analyze image frame for cheating   |
| GET    | `/docs`    | Interactive Swagger UI             |

## Deploy to Railway

1. Push this repo to GitHub
2. Create a new project on [Railway](https://railway.app)
3. Connect your GitHub repo
4. Railway auto-detects `railway.json` and builds the Dockerfile
5. Set `PORT` env var if needed (Railway injects it automatically)

## Local Development

```bash
pip install torch==2.2.0+cpu torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
