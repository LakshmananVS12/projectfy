"""
Road Damage Detection AI Service — FastAPI Application

Endpoints:
  POST /detect           — Synchronous photo detection (single image)
  POST /scan/video       — Submit async video scan job
  GET  /scan/{job_id}    — Poll video scan job status / fetch results
  GET  /health           — Health check
"""
from __future__ import annotations

import os
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference import RoadDamageInferenceService
from video_scan import DamageEvent, VideoJobManager, process_video

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEIGHTS_PATH = Path(os.getenv(
    "MODEL_WEIGHTS",
    str(Path(__file__).parent / "weights" / "hybrid_detector_best.pt"),
))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.30"))
NMS_IOU_THRESHOLD = float(os.getenv("NMS_IOU_THRESHOLD", "0.50"))
VIDEO_FPS_EXTRACT = float(os.getenv("VIDEO_FPS_EXTRACT", "1.0"))
VIDEO_TRACKER_IOU = float(os.getenv("VIDEO_TRACKER_IOU", "0.35"))
VIDEO_TRACKER_STALE = float(os.getenv("VIDEO_TRACKER_STALE", "2.0"))
MAX_VIDEO_WORKERS = int(os.getenv("MAX_VIDEO_WORKERS", "2"))

# Shared singletons — initialised at startup
inference_service: RoadDamageInferenceService | None = None
job_manager = VideoJobManager()
video_executor: ThreadPoolExecutor | None = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_service, video_executor

    inference_service = RoadDamageInferenceService(
        weights_path=WEIGHTS_PATH,
        image_size=IMAGE_SIZE,
        score_threshold=SCORE_THRESHOLD,
        nms_iou_threshold=NMS_IOU_THRESHOLD,
    )
    video_executor = ThreadPoolExecutor(max_workers=MAX_VIDEO_WORKERS)
    yield
    video_executor.shutdown(wait=False)


app = FastAPI(
    title="Road Damage Detection AI API",
    description="Hybrid CNN+ViT detection service for photos and videos.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# POST /detect — Synchronous single-image detection
# ---------------------------------------------------------------------------

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    """
    Accept a single road image, return detections + explainability heatmaps.

    Response shape:
    {
      "detections": [
        {"class": str, "bbox": [x1,y1,x2,y2], "confidence": float, "severity": str}
      ],
      "cnn_heatmap_base64": str,
      "vit_attention_base64": str,
      "processing_time_ms": float
    }
    """
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    result = inference_service.detect(image_bgr, include_explainability=True)

    return {
        "detections": result.detections,
        "cnn_heatmap_base64": result.cnn_heatmap_base64,
        "vit_attention_base64": result.vit_attention_base64,
        "processing_time_ms": result.processing_time_ms,
    }


# ---------------------------------------------------------------------------
# POST /scan/video — Submit async video job
# ---------------------------------------------------------------------------

def _run_video_job(job_id: str, video_path: str) -> None:
    """Background worker: process video and update job store."""
    try:
        job_manager.set_status(job_id, "PROCESSING")

        def _progress(pct: int):
            job_manager.set_progress(job_id, pct)

        events = process_video(
            video_path=video_path,
            inference_service=inference_service,
            fps_extract=VIDEO_FPS_EXTRACT,
            iou_threshold=VIDEO_TRACKER_IOU,
            stale_seconds=VIDEO_TRACKER_STALE,
            progress_callback=_progress,
        )

        results = [_event_to_dict(e) for e in events]
        job_manager.complete_job(job_id, results)

    except Exception as exc:
        job_manager.fail_job(job_id, str(exc))
        traceback.print_exc()
    finally:
        # Clean up temp video file
        try:
            os.unlink(video_path)
        except OSError:
            pass


def _event_to_dict(event: DamageEvent) -> Dict[str, Any]:
    return {
        "event_id": event.event_id,
        "damage_class": event.damage_class,
        "severity": event.severity,
        "confidence": round(event.confidence, 4),
        "timestamp_sec": round(event.timestamp_sec, 2),
        "bbox": [round(v, 1) for v in event.bbox],
        "representative_frame_base64": _frame_to_b64(event.representative_frame),
        "cnn_heatmap_base64": event.cnn_heatmap_base64,
        "vit_attention_base64": event.vit_attention_base64,
    }


def _frame_to_b64(frame: np.ndarray | None) -> str:
    if frame is None:
        return ""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    import base64
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.post("/scan/video")
async def scan_video(file: UploadFile = File(...)):
    """
    Submit a video for async damage scanning.

    Returns immediately with:
    { "job_id": str, "status": "QUEUED" }

    The backend (Spring Boot) should poll GET /scan/{job_id} for results.
    """
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if video_executor is None:
        raise HTTPException(status_code=503, detail="Video executor not ready.")

    # Save uploaded video to a temp file
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempfile.gettempdir())
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    job_id = job_manager.create_job()
    video_executor.submit(_run_video_job, job_id, tmp_path)

    return {"job_id": job_id, "status": "QUEUED"}


# ---------------------------------------------------------------------------
# GET /scan/{job_id} — Poll job status / fetch results
# ---------------------------------------------------------------------------

@app.get("/scan/{job_id}")
async def get_scan_status(job_id: str):
    """
    Poll a video scan job.

    Response shape:
    {
      "status": "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED",
      "progress_pct": int,           // 0-100
      "results": [...] | null,       // only when COMPLETED
      "error": str | null             // only when FAILED
    }

    Each result in the "results" array is a unique damage event:
    {
      "event_id": str,
      "damage_class": str,
      "severity": str,
      "confidence": float,
      "timestamp_sec": float,
      "bbox": [x1, y1, x2, y2],
      "representative_frame_base64": str,
      "cnn_heatmap_base64": str,
      "vit_attention_base64": str
    }
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    return {
        "status": job["status"],
        "progress_pct": job.get("progress_pct", 0),
        "results": job.get("results"),
        "error": job.get("error"),
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": inference_service is not None and inference_service._model is not None,
        "weights_path": str(WEIGHTS_PATH),
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
