from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from explainability import generate_explainability
from model import HybridDetectorConfig, build_hybrid_model, decode_detections
from severity import estimate_severity_for_detections


@dataclass
class InferenceResult:
    detections: List[Dict[str, object]]
    cnn_heatmap_base64: str
    vit_attention_base64: str
    processing_time_ms: float


class RoadDamageInferenceService:
    """Runtime inference helper for photo and video-frame detection.

    This loads the same hybrid model checkpoint and reuses it for both photo and
    video-frame inference (no separate video model).
    """

    def __init__(
        self,
        weights_path: Path,
        image_size: int = 640,
        score_threshold: float = 0.30,
        nms_iou_threshold: float = 0.50,
        device: str | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.image_size = image_size
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._model = None
        self._config: HybridDetectorConfig | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            if not self.weights_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {self.weights_path}")

            checkpoint = torch.load(self.weights_path, map_location="cpu")
            config_dict = dict(checkpoint.get("config") or {})

            if "class_names" in config_dict and isinstance(config_dict["class_names"], list):
                config_dict["class_names"] = tuple(config_dict["class_names"])

            # We already load trained weights from the checkpoint, so no backbone download is needed.
            config_dict["pretrained_backbones"] = False

            cfg = HybridDetectorConfig(**config_dict) if config_dict else HybridDetectorConfig(pretrained_backbones=False)
            model = build_hybrid_model(cfg)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.to(self.device)
            model.eval()

            self._config = cfg
            self._model = model

    def _preprocess(self, image_bgr: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input image is empty or invalid.")

        orig_h, orig_w = image_bgr.shape[:2]

        resized_bgr = cv2.resize(image_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = np.ascontiguousarray(resized_rgb)

        image_tensor = torch.from_numpy(resized_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.to(self.device)

        return image_tensor, resized_bgr, (orig_h, orig_w)

    def _scale_bbox_to_original(self, bbox: List[float], orig_hw: Tuple[int, int]) -> List[float]:
        orig_h, orig_w = orig_hw
        sx = orig_w / float(self.image_size)
        sy = orig_h / float(self.image_size)

        x1, y1, x2, y2 = bbox
        return [
            float(max(0.0, min(orig_w - 1.0, x1 * sx))),
            float(max(0.0, min(orig_h - 1.0, y1 * sy))),
            float(max(0.0, min(orig_w - 1.0, x2 * sx))),
            float(max(0.0, min(orig_h - 1.0, y2 * sy))),
        ]

    def _run_detection(self, image_tensor: torch.Tensor, orig_hw: Tuple[int, int]) -> List[Dict[str, object]]:
        assert self._model is not None
        assert self._config is not None

        with torch.no_grad():
            outputs = self._model(image_tensor)

        decoded = decode_detections(
            cls_logits=outputs["cls_logits"],
            bbox_reg=outputs["bbox_reg"],
            centerness=outputs["centerness"],
            stride=self._model.stride,
            class_names=self._config.class_names,
            score_threshold=self.score_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
            top_k=200,
            image_hw=(self.image_size, self.image_size),
        )

        detections = decoded[0] if decoded else []
        for det in detections:
            det["bbox"] = self._scale_bbox_to_original(det["bbox"], orig_hw=orig_hw)

        detections = estimate_severity_for_detections(detections, image_width=orig_hw[1])
        return detections

    def detect_detections_only(self, image_bgr: np.ndarray) -> List[Dict[str, object]]:
        self._ensure_loaded()
        with self._inference_lock:
            image_tensor, _resized_bgr, orig_hw = self._preprocess(image_bgr)
            return self._run_detection(image_tensor=image_tensor, orig_hw=orig_hw)

    def generate_frame_heatmaps(self, image_bgr: np.ndarray) -> Dict[str, str]:
        self._ensure_loaded()
        assert self._model is not None

        with self._inference_lock:
            image_tensor, resized_bgr, _orig_hw = self._preprocess(image_bgr)
            heatmaps = generate_explainability(
                model=self._model,
                image_tensor=image_tensor,
                image_bgr=resized_bgr,
            )

        return {
            "cnn_heatmap_base64": heatmaps.cnn_heatmap_base64,
            "vit_attention_base64": heatmaps.vit_attention_base64,
        }

    def detect(self, image_bgr: np.ndarray, include_explainability: bool = True) -> InferenceResult:
        self._ensure_loaded()
        assert self._model is not None

        start = perf_counter()
        with self._inference_lock:
            image_tensor, resized_bgr, orig_hw = self._preprocess(image_bgr)
            detections = self._run_detection(image_tensor=image_tensor, orig_hw=orig_hw)

            cnn_heatmap = ""
            vit_heatmap = ""
            if include_explainability:
                heatmaps = generate_explainability(
                    model=self._model,
                    image_tensor=image_tensor,
                    image_bgr=resized_bgr,
                )
                cnn_heatmap = heatmaps.cnn_heatmap_base64
                vit_heatmap = heatmaps.vit_attention_base64

        processing_ms = (perf_counter() - start) * 1000.0

        cleaned = [
            {
                "class": det["class"],
                "bbox": det["bbox"],
                "confidence": float(det["confidence"]),
                "severity": det.get("severity", "LOW"),
            }
            for det in detections
        ]

        return InferenceResult(
            detections=cleaned,
            cnn_heatmap_base64=cnn_heatmap,
            vit_attention_base64=vit_heatmap,
            processing_time_ms=round(processing_ms, 2),
        )
