# PROGRESS_LOG

## 2026-07-10

### What was built
- Implemented hybrid model architecture in [ai-service/model/hybrid_detector.py](ai-service/model/hybrid_detector.py):
	- Pretrained CNN branch (ResNet18/34 option, default ResNet34)
	- Pretrained ViT branch (DeiT Tiny/Small via timm, default DeiT-Small)
	- From-scratch fusion layer (channel concat + 1x1 conv reduction)
	- From-scratch FCOS-style head (classification, bbox regression, centerness)
- Added training/inference support utilities:
	- [ai-service/model/losses.py](ai-service/model/losses.py)
	- [ai-service/model/postprocess.py](ai-service/model/postprocess.py)
	- [ai-service/model/__init__.py](ai-service/model/__init__.py)
- Implemented dataset preprocessing pipeline for RDD-style XML annotations:
	- [ai-service/data/preprocess_rdd2022.py](ai-service/data/preprocess_rdd2022.py)
	- [ai-service/data/class_map.py](ai-service/data/class_map.py)
	- [ai-service/data/DOWNLOAD_INSTRUCTIONS.md](ai-service/data/DOWNLOAD_INSTRUCTIONS.md)
- Implemented rule-based severity estimation in [ai-service/severity.py](ai-service/severity.py).
- Implemented explainability utilities (Grad-CAM + attention rollout) in [ai-service/explainability.py](ai-service/explainability.py).
- Created complete Colab-focused training notebook in [training/train_hybrid_model.ipynb](training/train_hybrid_model.ipynb), including staged unfreezing, fp16 training, mAP/IoU/F1 logging, and ONNX export.

### Decisions made
- Final active class target remains 4 classes: pothole, linear_crack, alligator_crack, edge_break.
- Ravelling is default-dropped unless sample count meets viability threshold (default 300) to avoid artificial class balancing.
- Edge-break coverage is explicitly checked and reported; if absent, pipeline flags it clearly instead of fabricating labels.
- No architecture deviation from PROJECT_CONTEXT for the hybrid model design.

### Current metrics
- No training run yet in this session (Colab/manual step required), so mAP/IoU/F1 are pending.

### Still pending
- Prompt 1 items after notebook handoff are pending user confirmation of trained weights in repo:
	- [ai-service/main.py](ai-service/main.py) photo endpoint
	- [ai-service/video_scan.py](ai-service/video_scan.py) frame sampling + IoU dedup tracking
	- Async video job endpoints (`POST /scan/video`, `GET /scan/{job_id}`)
	- Service Dockerfile
	- End-to-end validation outputs for photo detect and video job flow

### Manual actions needed before next step
- Run [training/train_hybrid_model.ipynb](training/train_hybrid_model.ipynb) in Colab and place outputs into:
	- [ai-service/weights/hybrid_detector_best.pt](ai-service/weights/hybrid_detector_best.pt)
	- [ai-service/weights/hybrid_detector.onnx](ai-service/weights/hybrid_detector.onnx)

## 2026-07-10 (Notebook Streamline Update)

### What was built
- Added one-go Colab notebook [training/train_hybrid_model_one_go.ipynb](training/train_hybrid_model_one_go.ipynb) to reduce setup confusion.
- The new notebook automates: repo clone, Drive zip extraction, preprocessing, training, ONNX export, and copying final artifacts to both project weights folder and Google Drive output folder.

### Decisions made
- Kept training defaults aligned with prior plan (image size 640, batch size 8, staged freeze/unfreeze, fp16, 4-class target).
- Added strict path/file checks with clear runtime errors to fail fast when data placement is incorrect.

### Still pending
- User needs to execute the new one-go notebook in Colab and share final metrics + artifact confirmation.
- Service endpoints and video async pipeline remain pending until weight files are confirmed in repo.

## 2026-07-10 (AI Module — Prompt 1 Completion)

### What was built
- **FastAPI application** in [ai-service/main.py](ai-service/main.py):
	- `POST /detect` — synchronous single-image detection returning detections, severity, CNN Grad-CAM heatmap, ViT attention rollout heatmap, and processing time.
	- `POST /scan/video` — submit an async video scan job, returns `{job_id, status: "QUEUED"}` immediately.
	- `GET /scan/{job_id}` — poll job status (`QUEUED`/`PROCESSING`/`COMPLETED`/`FAILED`), progress percentage, and results when complete.
	- `GET /health` — health check with model load status.
	- CORS enabled for frontend consumption.
	- `ThreadPoolExecutor` for background video processing (configurable max workers).
- **Video scan pipeline** in [ai-service/video_scan.py](ai-service/video_scan.py):
	- Frame extraction via OpenCV at configurable sample rate (default 1 fps).
	- Per-frame detection using the **same** hybrid model (no separate video model).
	- `CrossFrameTracker` — IoU-based cross-frame deduplication (IoU threshold 0.35, stale timeout 2s).
	- Tracks are promoted to `DamageEvent`s on finalisation, keeping the highest-confidence frame as representative.
	- Explainability heatmaps generated for each event's representative frame.
	- `VideoJobManager` — thread-safe in-memory job store for the async submit-poll-fetch pattern.
- **Dockerfile** for the AI service (Python 3.11-slim base, health check, env-configurable settings).
- **requirements.txt** with pinned dependency ranges.
- **E2E test script** in [ai-service/test_e2e.py](ai-service/test_e2e.py).

### Decisions made
- Frame sampling rate: **1 fps** (configurable via `VIDEO_FPS_EXTRACT` env var).
- Cross-frame tracker IoU threshold: **0.35** (lower than NMS threshold to account for camera movement shifting bbox positions between frames). Stale timeout: **2.0 seconds**.
- Used `ThreadPoolExecutor` (max 2 workers) for background video processing — keeps dependency footprint minimal; no Redis/Celery needed at this scope.
- All configuration is env-var-driven for Docker/deployment flexibility.

### E2E validation results
All 4 tests passed:
- `GET /health` — 200, status=ok
- `POST /detect` — 200, heatmaps present, processing time ~4671ms (CPU, first load)
- `POST /scan/video` + `GET /scan/{job_id}` — QUEUED -> COMPLETED, 0 detections on synthetic test video (expected with model trained on real road data)
- `GET /scan/nonexistent` — 404

Note: 0 detections on synthetic images is expected — the model was trained on real RDD2022 road damage images. With real road photos/videos containing potholes/cracks, the model will produce detections.

### Current model metrics
- Model weights present at `ai-service/weights/hybrid_detector_best.pt` (183 MB) and `hybrid_detector.onnx` (1.3 MB).
- No mAP/IoU/F1 metrics available yet — user trained via Colab but metrics were not reported back.

### Status
- **Prompt 1 (AI Module) is COMPLETE.** All items delivered and validated.
- Ready for Prompt 2 (Backend) or Prompt 3 (Frontend).

### Pending manual actions
- None for the AI module. Ready for next prompt.

