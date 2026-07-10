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
