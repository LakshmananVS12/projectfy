"""
Video scan pipeline: frame extraction, per-frame detection, cross-frame deduplication.

Uses the SAME hybrid model checkpoint used for photo detection -- no separate video model.
"""
from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from inference import RoadDamageInferenceService


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DamageEvent:
    """A unique damage event consolidated across multiple video frames."""
    event_id: str
    damage_class: str
    severity: str
    confidence: float
    timestamp_sec: float
    bbox: List[float]
    representative_frame: np.ndarray | None = field(default=None, repr=False)
    cnn_heatmap_base64: str = ""
    vit_attention_base64: str = ""


@dataclass
class _TrackedDetection:
    """Internal state for an active track being followed across frames."""
    event_id: str
    damage_class: str
    bbox: List[float]
    best_confidence: float
    best_timestamp: float
    best_frame: np.ndarray | None
    severity: str
    last_seen_timestamp: float
    hit_count: int = 1


# ---------------------------------------------------------------------------
# IoU-based cross-frame tracker
# ---------------------------------------------------------------------------

def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class CrossFrameTracker:
    """
    Simple IoU-based tracker that merges detections of the same class
    with high bounding-box overlap across consecutive frames into a single
    "damage event".

    Parameters:
        iou_threshold: minimum IoU to match a detection to an existing track.
        stale_seconds: seconds without a match before a track is finalised.
    """

    def __init__(self, iou_threshold: float = 0.35, stale_seconds: float = 2.0) -> None:
        self.iou_threshold = iou_threshold
        self.stale_seconds = stale_seconds
        self._active_tracks: List[_TrackedDetection] = []
        self._completed: List[_TrackedDetection] = []

    def update(
        self,
        detections: List[Dict[str, Any]],
        timestamp_sec: float,
        frame_bgr: np.ndarray | None = None,
    ) -> None:
        """Feed detections from one frame into the tracker."""
        matched_track_ids: set = set()

        for det in detections:
            det_class = det["class"]
            det_bbox = det["bbox"]
            det_conf = float(det["confidence"])
            det_sev = det.get("severity", "LOW")

            best_iou = 0.0
            best_track_idx = -1

            for idx, track in enumerate(self._active_tracks):
                if idx in matched_track_ids:
                    continue
                if track.damage_class != det_class:
                    continue

                iou = _compute_iou(track.bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = idx

            if best_iou >= self.iou_threshold and best_track_idx >= 0:
                # Update existing track
                track = self._active_tracks[best_track_idx]
                track.bbox = det_bbox
                track.last_seen_timestamp = timestamp_sec
                track.hit_count += 1
                if det_conf > track.best_confidence:
                    track.best_confidence = det_conf
                    track.best_timestamp = timestamp_sec
                    track.best_frame = frame_bgr.copy() if frame_bgr is not None else None
                    track.severity = det_sev
                matched_track_ids.add(best_track_idx)
            else:
                # Start a new track
                self._active_tracks.append(
                    _TrackedDetection(
                        event_id=uuid.uuid4().hex[:12],
                        damage_class=det_class,
                        bbox=det_bbox,
                        best_confidence=det_conf,
                        best_timestamp=timestamp_sec,
                        best_frame=frame_bgr.copy() if frame_bgr is not None else None,
                        severity=det_sev,
                        last_seen_timestamp=timestamp_sec,
                    )
                )

        # Retire stale tracks
        still_active: List[_TrackedDetection] = []
        for track in self._active_tracks:
            if timestamp_sec - track.last_seen_timestamp > self.stale_seconds:
                self._completed.append(track)
            else:
                still_active.append(track)
        self._active_tracks = still_active

    def finalise(self) -> List[DamageEvent]:
        """Flush all remaining tracks and return deduplicated damage events."""
        all_tracks = self._completed + self._active_tracks
        self._active_tracks = []
        self._completed = []

        events: List[DamageEvent] = []
        for track in all_tracks:
            frame_b64 = ""
            heatmap_cnn = ""
            heatmap_vit = ""
            if track.best_frame is not None:
                ok, buf = cv2.imencode(".jpg", track.best_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    frame_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            events.append(
                DamageEvent(
                    event_id=track.event_id,
                    damage_class=track.damage_class,
                    severity=track.severity,
                    confidence=track.best_confidence,
                    timestamp_sec=track.best_timestamp,
                    bbox=track.bbox,
                    representative_frame=track.best_frame,
                    cnn_heatmap_base64=heatmap_cnn,
                    vit_attention_base64=heatmap_vit,
                )
            )
        return events


# ---------------------------------------------------------------------------
# Video processing pipeline
# ---------------------------------------------------------------------------

def process_video(
    video_path: str | Path,
    inference_service: RoadDamageInferenceService,
    fps_extract: float = 1.0,
    iou_threshold: float = 0.35,
    stale_seconds: float = 2.0,
    progress_callback=None,
) -> List[DamageEvent]:
    """
    Full video scan pipeline:
      1. Frame extraction at configurable sample rate (default 1 fps).
      2. Per-frame detection via the same hybrid model.
      3. Cross-frame deduplication via IoU tracker.
      4. Generate explainability heatmaps for each event's representative frame.

    Args:
        video_path: path to the video file.
        inference_service: shared RoadDamageInferenceService instance.
        fps_extract: how many frames per second to sample.
        iou_threshold: IoU threshold for cross-frame matching.
        stale_seconds: seconds without a match before a track is finalised.
        progress_callback: optional callable(pct: int) for progress updates.

    Returns:
        List of unique DamageEvents.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps != video_fps:  # NaN check
        video_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(video_fps / fps_extract)))

    tracker = CrossFrameTracker(iou_threshold=iou_threshold, stale_seconds=stale_seconds)

    frame_idx = 0
    processed_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_sec = frame_idx / video_fps

                # Run the same model used for photo detection
                detections = inference_service.detect_detections_only(frame)
                tracker.update(detections, timestamp_sec, frame_bgr=frame)
                processed_count += 1

                if progress_callback and total_frames > 0:
                    pct = min(99, int(frame_idx / total_frames * 100))
                    progress_callback(pct)

            frame_idx += 1
    finally:
        cap.release()

    events = tracker.finalise()

    # Generate explainability heatmaps for each event's best frame
    for event in events:
        if event.representative_frame is not None:
            try:
                heatmaps = inference_service.generate_frame_heatmaps(event.representative_frame)
                event.cnn_heatmap_base64 = heatmaps["cnn_heatmap_base64"]
                event.vit_attention_base64 = heatmaps["vit_attention_base64"]
            except Exception:
                pass  # Heatmap generation failure shouldn't crash the pipeline

    return events


# ---------------------------------------------------------------------------
# Async job management
# ---------------------------------------------------------------------------

class VideoJobManager:
    """
    In-memory job store for async video scan processing.

    Job lifecycle: QUEUED -> PROCESSING -> COMPLETED | FAILED
    The Spring Boot backend (Prompt 3) will poll GET /scan/{job_id}.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def create_job(self) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "status": "QUEUED",
                "progress_pct": 0,
                "results": None,
                "error": None,
            }
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_id)

    def set_status(self, job_id: str, status: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = status

    def set_progress(self, job_id: str, pct: int) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["progress_pct"] = pct

    def complete_job(self, job_id: str, results: List[Dict]) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "COMPLETED"
                self._jobs[job_id]["progress_pct"] = 100
                self._jobs[job_id]["results"] = results

    def fail_job(self, job_id: str, error: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "FAILED"
                self._jobs[job_id]["error"] = error
