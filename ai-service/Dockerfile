# ---- Stage 1: Base image with system deps ----
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Stage 2: Install Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Stage 3: Copy application code ----
COPY . .

# Expose API port
EXPOSE 8000

# Environment defaults (overridable at runtime)
ENV MODEL_WEIGHTS=/app/weights/hybrid_detector_best.pt \
    IMAGE_SIZE=640 \
    SCORE_THRESHOLD=0.30 \
    NMS_IOU_THRESHOLD=0.50 \
    VIDEO_FPS_EXTRACT=1.0 \
    VIDEO_TRACKER_IOU=0.35 \
    VIDEO_TRACKER_STALE=2.0 \
    MAX_VIDEO_WORKERS=2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
