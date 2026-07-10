# PROJECT_CONTEXT.md

## Read this file in full before starting work in ANY session, on ANY module.

This file is the single source of truth for the whole project. Every module prompt in this folder will tell you to re-read this file first. If anything in a module prompt seems to conflict with this file, this file wins -- flag the conflict to me instead of guessing.

---

## What we're building

An AI-powered road damage detection and maintenance management platform. Citizens/inspectors upload geotagged road photos or videos of a road (e.g. a dashcam-style drive-through). A hybrid CNN+ViT model detects damage in photos, and for videos, runs frame-by-frame across the clip and consolidates repeated detections of the same damage into single events. It localizes damage, estimates severity, and explains its reasoning visually. A Spring Boot backend runs the full report lifecycle, computes health/priority/risk scores, and recommends repairs. A React frontend gives five different roles five different tailored experiences. It must look and function like a professional civic-tech product.

## Fixed tech stack -- never substitute without asking me

| Layer | Stack |
| --- | --- |
| Frontend | React, Leaflet, Recharts |
| Backend | Spring Boot, Spring Security (JWT), Spring Data JPA |
| AI Module | Python, FastAPI, PyTorch, timm, OpenCV |
| Database | PostgreSQL + PostGIS |
| Explainability | Grad-CAM (CNN branch) + attention rollout (ViT branch) |

## The AI architecture -- this is the core technical identity of the project, get it right

A genuine two-branch hybrid, not a single-model substitute:

- CNN branch: pretrained ResNet18/34, feature extractor for local texture/edges (fine cracks).
- ViT branch: pretrained DeiT-Tiny/Small (via `timm`), processes the full image for global context (tells real damage apart from shadows/debris/tar patches).
- Fusion: ViT tokens reshaped to a spatial grid, concatenated with CNN feature map channels, reduced via 1x1 conv.
- Detection head: lightweight FCOS-style anchor-free head, trained from scratch on fused features -- classification (4 classes) + bbox regression + centerness.
- Damage classes: `pothole`, `linear_crack` (merged longitudinal/transverse), `alligator_crack`, `edge_break`. `ravelling` only included if RDD2022 has enough samples -- otherwise flagged and dropped, not faked.
- Explainability: Grad-CAM for CNN branch, attention rollout for ViT branch, both returned per detection.
- Training happens on Google Colab (free T4 GPU) -- local GPU is an RTX 3050 4GB, too small for this model. Model exported to ONNX after training.

## Video input support -- runs the SAME trained model over frames, not a separate model

The AI module must also accept a video of a road (e.g. filmed while walking or driving) and detect damage across it. This does NOT require training a new/different model -- the same hybrid image detector is applied per-frame. What's new is the pipeline around it:

- Frame extraction: sample frames at a configurable interval (e.g. 1-2 frames per second, not every single frame -- full frame-rate processing is unnecessary and slow).
- Per-frame detection: run the existing hybrid model on each sampled frame, exactly as for a photo.
- Cross-frame deduplication (tracking): the same pothole will appear in many consecutive frames as the camera moves past it. Use a simple IoU-based tracker (match detections of the same class with high bounding-box overlap across consecutive frames) to merge these into a single damage event rather than creating duplicate reports for the same pothole seen 10 times.
- Output: a list of unique damage events, each with: damage type, severity, confidence, a representative frame (best/clearest detection), approximate timestamp in the video, and explainability heatmaps for that representative frame.
- Processing is asynchronous: video processing takes meaningfully longer than a single photo (many frames to run inference on). This must be a submit-job -> poll-status -> fetch-results pattern, not a single blocking HTTP request. Design the API this way from the start.
- Video-based scanning is primarily an Inspector tool (driving/walking a road and recording it) but the architecture should not hard-block Citizens from using it if they have a video, since the core spec allows either role to submit.

## Full feature scope -- what's in, what's simplified

| Feature | Status | Note |
| --- | --- | --- |
| Multi-class damage detection + bbox (photo) | IN | 4 classes, see above |
| Multi-class damage detection from video | IN | Frame sampling + per-frame detection + cross-frame dedup, async job pattern |
| Severity estimation | IN | Rule-based on bbox area vs road width + damage type |
| Repair Priority Score | IN | Backend formula: severity + road category + traffic importance -- ranks individual reports/events |
| Road Health Score | IN | Formula includes damage count, severity, repair recency, and inspection staleness (4 inputs) -- ranks road segments |
| Predictive Maintenance | IN (simplified) | Rule-based deterioration-risk flag (Low/Med/High), explicitly labeled as "Phase 1, upgradeable to ML later" |
| AI Repair Recommendation | IN (simplified) | Rule + admin-editable lookup table for method/cost-range/duration, not a trained regression model |
| Full lifecycle state machine | IN | Reported -> Verified -> Assigned -> In-Progress -> Completed |
| Interactive map + dashboard | IN | Leaflet + Recharts |
| Segmentation masks | STRETCH | Only after bbox pipeline is solid |

## Database schema (PostgreSQL + PostGIS) -- do not redesign without flagging me

- `users` (id, name, email, password_hash, role)
- `road_segments` (id, geometry [PostGIS LineString], name, road_category, traffic_importance, current_health_score, deterioration_risk, last_inspected_at)
- `reports` (id, road_segment_id FK, reporter_id FK, source_type [PHOTO or VIDEO_FRAME], scan_job_id FK nullable, image_url, location [PostGIS Point], damage_type, severity, confidence, priority_score, status, created_at, verified_by, verified_at)
- `repairs` (id, report_id FK, contractor_id FK, method, estimated_cost, estimated_duration, status, completion_photo_url, completed_at)
- `repair_lookup` (damage_type, severity, method, estimated_cost_range, estimated_duration)
- `scan_jobs` (id, road_segment_id FK, submitted_by FK, video_url, status [QUEUED/PROCESSING/COMPLETED/FAILED], submitted_at, completed_at) -- a video scan produces one `scan_job` and multiple resulting `reports` (one per unique damage event), each linked back via `scan_job_id`.

## The five roles -- never blur these

- CITIZEN: report issues (photo or video), view own reports + public dashboard. Nothing else.
- INSPECTOR: everything Citizen has + bulk field photo upload + road video scan (primary use case for video) + read-only segment health view.
- ENGINEER: verification queue (covers both photo and video-derived reports), priority queue, segment health view, repair recommendation editing. Only role that verifies.
- ADMIN: system dashboard, contractor assignment, user management, repair_lookup table editor. Only role with full visibility + repair_lookup edit rights.
- CONTRACTOR: assigned repairs list, status updates, completion photo upload. Only role that marks Completed.

## Design/UX standard

Read `skill.md` in the repo root before any UI work -- apply its token-based design process (color/type/layout/signature). This is a civic-infrastructure product for government staff, contractors, and the public -- must look trustworthy and professional, not like a student CRUD scaffold. Real empty/loading/error states everywhere, including a proper processing state for video scans (this can take minutes -- treat it as a first-class async UX pattern, not an afterthought spinner). Role-specific navigation (not one shared sidebar with hidden items). Route-level role guarding, not just hidden nav links.

## Agent behavior rules -- apply in every session

You cannot run cloud GPU training, click through Colab, create accounts, download gated datasets, or move files onto my local machine outside this repo. Whenever you hit one of these, STOP and give me an explicit numbered instruction block (what to open, click, run, download, and where to place the result back in the repo). Never assume I did a manual step -- ask me to confirm before writing code that depends on it.

## Continuity rule -- do this at the START and END of every module session

1. At the start: re-read this file AND `PROGRESS_LOG.md` (if it exists) before writing any code.
2. At the end of the session: append a dated entry to `PROGRESS_LOG.md` summarizing what you built, what decisions you made, what's still pending, and anything I need to do manually before the next session. Create `PROGRESS_LOG.md` if it doesn't exist yet.
