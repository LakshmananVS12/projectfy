from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torchvision.ops import nms


def _locations(feature_h: int, feature_w: int, stride: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(0, feature_h, device=device, dtype=torch.float32)
    xs = torch.arange(0, feature_w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cx = (xx + 0.5) * stride
    cy = (yy + 0.5) * stride
    return torch.stack([cx, cy], dim=-1)


def decode_detections(
    cls_logits: torch.Tensor,
    bbox_reg: torch.Tensor,
    centerness: torch.Tensor,
    stride: int,
    class_names: Sequence[str],
    score_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
    top_k: int = 200,
    image_hw: Tuple[int, int] | None = None,
) -> List[List[Dict[str, object]]]:
    """Decode FCOS-style output maps into final detections with per-image NMS."""

    batch, num_classes, feat_h, feat_w = cls_logits.shape
    centers = _locations(feat_h, feat_w, stride, cls_logits.device)

    cls_probs = torch.sigmoid(cls_logits)
    center_probs = torch.sigmoid(centerness)

    batch_results: List[List[Dict[str, object]]] = []
    for b_idx in range(batch):
        score_map = cls_probs[b_idx] * center_probs[b_idx]
        score_map = score_map.reshape(num_classes, -1)

        boxes_per_class: List[torch.Tensor] = []
        scores_per_class: List[torch.Tensor] = []
        labels_per_class: List[torch.Tensor] = []

        reg = bbox_reg[b_idx].permute(1, 2, 0).reshape(-1, 4)
        center_flat = centers.reshape(-1, 2)

        for cls_id in range(num_classes):
            cls_scores = score_map[cls_id]
            keep = cls_scores > score_threshold
            if keep.sum() == 0:
                continue

            cls_scores = cls_scores[keep]
            cls_reg = reg[keep]
            cls_center = center_flat[keep]

            if cls_scores.numel() > top_k:
                top_vals, top_idx = torch.topk(cls_scores, top_k)
                cls_scores = top_vals
                cls_reg = cls_reg[top_idx]
                cls_center = cls_center[top_idx]

            x1 = cls_center[:, 0] - cls_reg[:, 0]
            y1 = cls_center[:, 1] - cls_reg[:, 1]
            x2 = cls_center[:, 0] + cls_reg[:, 2]
            y2 = cls_center[:, 1] + cls_reg[:, 3]

            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            boxes_per_class.append(boxes)
            scores_per_class.append(cls_scores)
            labels_per_class.append(torch.full_like(cls_scores, fill_value=cls_id, dtype=torch.long))

        if not boxes_per_class:
            batch_results.append([])
            continue

        boxes_all = torch.cat(boxes_per_class, dim=0)
        scores_all = torch.cat(scores_per_class, dim=0)
        labels_all = torch.cat(labels_per_class, dim=0)

        if image_hw is not None:
            img_h, img_w = image_hw
            boxes_all[:, 0::2] = boxes_all[:, 0::2].clamp(min=0.0, max=float(img_w - 1))
            boxes_all[:, 1::2] = boxes_all[:, 1::2].clamp(min=0.0, max=float(img_h - 1))

        keep_idx = nms(boxes_all, scores_all, iou_threshold=nms_iou_threshold)
        keep_idx = keep_idx[:top_k]

        image_results: List[Dict[str, object]] = []
        for idx in keep_idx:
            cls_id = int(labels_all[idx].item())
            image_results.append(
                {
                    "class_id": cls_id,
                    "class": class_names[cls_id] if cls_id < len(class_names) else str(cls_id),
                    "confidence": float(scores_all[idx].item()),
                    "bbox": [float(v) for v in boxes_all[idx].tolist()],
                }
            )

        batch_results.append(image_results)

    return batch_results
