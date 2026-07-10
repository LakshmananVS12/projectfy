from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    modulating = (1.0 - p_t).pow(gamma)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * modulating * ce_loss


class FCOSLoss(nn.Module):
    """Simplified FCOS-style losses for single-level training."""

    def __init__(self, num_classes: int, stride: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.stride = stride

    def _build_targets(
        self,
        gt_boxes_per_image: List[torch.Tensor],
        feat_h: int,
        feat_w: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        batch = len(gt_boxes_per_image)
        cls_targets = torch.zeros(batch, self.num_classes, feat_h, feat_w, device=device)
        reg_targets = torch.zeros(batch, 4, feat_h, feat_w, device=device)
        center_targets = torch.zeros(batch, 1, feat_h, feat_w, device=device)
        positive_mask = torch.zeros(batch, 1, feat_h, feat_w, dtype=torch.bool, device=device)

        for b_idx, gt in enumerate(gt_boxes_per_image):
            if gt.numel() == 0:
                continue

            for row in gt:
                x1, y1, x2, y2, cls_id = row.tolist()
                cls_id = int(cls_id)
                if cls_id < 0 or cls_id >= self.num_classes:
                    continue

                cx = 0.5 * (x1 + x2) / self.stride
                cy = 0.5 * (y1 + y2) / self.stride
                ix = min(max(int(round(cx)), 0), feat_w - 1)
                iy = min(max(int(round(cy)), 0), feat_h - 1)

                px = (ix + 0.5) * self.stride
                py = (iy + 0.5) * self.stride
                l = max(px - x1, 1e-6)
                t = max(py - y1, 1e-6)
                r = max(x2 - px, 1e-6)
                btm = max(y2 - py, 1e-6)

                cls_targets[b_idx, cls_id, iy, ix] = 1.0
                # Normalize regression targets by stride for stable gradients
                reg_targets[b_idx, :, iy, ix] = torch.tensor([l, t, r, btm], device=device) / self.stride
                center_targets[b_idx, 0, iy, ix] = ((min(l, r) / max(l, r)) * (min(t, btm) / max(t, btm))) ** 0.5
                positive_mask[b_idx, 0, iy, ix] = True

        return {
            "cls": cls_targets,
            "reg": reg_targets,
            "center": center_targets,
            "mask": positive_mask,
        }

    def forward(
        self,
        cls_logits: torch.Tensor,
        bbox_reg: torch.Tensor,
        centerness: torch.Tensor,
        gt_boxes_per_image: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        bsz, _, feat_h, feat_w = cls_logits.shape
        targets = self._build_targets(gt_boxes_per_image, feat_h, feat_w, cls_logits.device)

        # 1. Focal Loss for Classification
        # Sum over all pixels, but normalize by the number of positive samples (boxes)
        focal_loss_map = _sigmoid_focal_loss(cls_logits, targets["cls"])
        num_pos = targets["mask"].sum().clamp(min=1.0)
        cls_loss = focal_loss_map.sum() / num_pos

        reg_loss = torch.tensor(0.0, device=cls_logits.device)
        center_loss = torch.tensor(0.0, device=cls_logits.device)

        if targets["mask"].any():
            # 2. Regression Loss
            pos_mask_reg = targets["mask"].expand(-1, 4, -1, -1)
            # Normalize bbox_reg by stride as well, since targets are normalized
            # We use Smooth L1 on the normalized distances, which makes the loss much smaller and bounded
            bbox_reg_norm = bbox_reg / self.stride
            
            # Multiply by centerness targets to downweight poor bounding boxes (standard FCOS trick)
            center_weights = targets["center"][targets["mask"]].unsqueeze(1)
            
            # Compute Smooth L1 (shape: [num_pos, 4])
            per_box_reg_loss = F.smooth_l1_loss(
                bbox_reg_norm[pos_mask_reg].view(-1, 4),
                targets["reg"][pos_mask_reg].view(-1, 4),
                reduction="none"
            )
            # Apply centerness weights and sum, then average over num_pos
            reg_loss = (per_box_reg_loss * center_weights).sum() / num_pos

            # 3. Centerness Loss
            center_loss = F.binary_cross_entropy_with_logits(
                centerness[targets["mask"]],
                targets["center"][targets["mask"]],
                reduction="sum",
            ) / num_pos

        total = cls_loss + reg_loss + center_loss
        return {
            "loss_total": total,
            "loss_cls": cls_loss.detach(),
            "loss_reg": reg_loss.detach(),
            "loss_center": center_loss.detach(),
        }
