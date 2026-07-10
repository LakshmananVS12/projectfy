from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import base64
import io

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ExplainabilityResult:
    cnn_heatmap_base64: str
    vit_attention_base64: str


def _normalize_map(attn_map: np.ndarray) -> np.ndarray:
    attn_map = np.maximum(attn_map, 0)
    max_val = float(attn_map.max()) if attn_map.size else 0.0
    if max_val <= 1e-8:
        return np.zeros_like(attn_map, dtype=np.float32)
    return (attn_map / max_val).astype(np.float32)


def _overlay_to_base64(
    heatmap_2d: np.ndarray,
    image_bgr: np.ndarray,
    alpha: float = 0.45,
) -> str:
    h, w = image_bgr.shape[:2]
    heatmap_2d = cv2.resize(heatmap_2d, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_u8 = np.uint8(np.clip(heatmap_2d, 0, 1) * 255)
    color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image_bgr, 1 - alpha, color, alpha, 0)

    ok, encoded = cv2.imencode(".png", blended)
    if not ok:
        raise RuntimeError("Failed to encode heatmap image.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class GradCAM:
    """Grad-CAM over the CNN branch of the hybrid detector."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        target_layer = model.cnn_branch.layer3[-1]
        self._fwd = target_layer.register_forward_hook(self._forward_hook)
        self._bwd = target_layer.register_full_backward_hook(self._backward_hook)

    def close(self) -> None:
        self._fwd.remove()
        self._bwd.remove()

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: int | None = None,
        fmap_xy: Tuple[int, int] | None = None,
    ) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(image_tensor)

        cls_logits = outputs["cls_logits"]
        centerness = outputs["centerness"]
        score_map = torch.sigmoid(cls_logits) * torch.sigmoid(centerness)

        if class_idx is None or fmap_xy is None:
            flat_idx = torch.argmax(score_map[0]).item()
            num_classes, h, w = score_map.shape[1:]
            class_idx = flat_idx // (h * w)
            rem = flat_idx % (h * w)
            y = rem // w
            x = rem % w
        else:
            x, y = fmap_xy

        target_score = cls_logits[0, class_idx, y, x] + centerness[0, 0, y, x]
        target_score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activations = self.activations[0]
        gradients = self.gradients[0]

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=0, keepdim=True))
        cam = F.interpolate(
            cam.unsqueeze(0),
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        return _normalize_map(cam.detach().cpu().numpy())


class AttentionRollout:
    """Attention rollout for the ViT branch.

    Uses forward-pre hooks to reconstruct attention matrices from qkv weights during
    a standard forward pass.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def _collect_attentions(self, image_tensor: torch.Tensor) -> List[torch.Tensor]:
        attentions: List[torch.Tensor] = []
        hooks = []

        def pre_hook(module, inputs):
            x = inputs[0]
            bsz, token_count, channels = x.shape
            num_heads = module.num_heads
            head_dim = channels // num_heads

            qkv = F.linear(x, module.qkv.weight, module.qkv.bias)
            qkv = qkv.reshape(bsz, token_count, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            attentions.append(attn.detach())

        for block in self.model.vit_branch.blocks:
            hooks.append(block.attn.register_forward_pre_hook(pre_hook))

        with torch.no_grad():
            _ = self.model.vit_branch.forward_features(image_tensor)

        for h in hooks:
            h.remove()

        return attentions

    def generate(self, image_tensor: torch.Tensor) -> np.ndarray:
        attentions = self._collect_attentions(image_tensor)
        if not attentions:
            raise RuntimeError("No ViT attention maps were captured.")

        bsz, _, token_count, _ = attentions[0].shape
        eye = torch.eye(token_count, device=image_tensor.device).unsqueeze(0).expand(bsz, -1, -1)
        rollout = eye.clone()

        for attn in attentions:
            attn_mean = attn.mean(dim=1)
            attn_mean = 0.5 * attn_mean + 0.5 * eye
            attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            rollout = attn_mean @ rollout

        img_h, img_w = image_tensor.shape[-2:]
        patch_h, patch_w = img_h // 16, img_w // 16
        patch_tokens = patch_h * patch_w
        prefix_tokens = max(0, token_count - patch_tokens)

        if prefix_tokens > 0:
            # Use CLS-token rollout when prefix tokens exist.
            mask = rollout[0, 0, prefix_tokens:]
        else:
            # Fallback to mean attention if no CLS token exists.
            mask = rollout[0].mean(dim=0)

        mask = mask.reshape(patch_h, patch_w).unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=(img_h, img_w), mode="bilinear", align_corners=False)
        return _normalize_map(mask[0, 0].detach().cpu().numpy())


def generate_explainability(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    image_bgr: np.ndarray,
) -> ExplainabilityResult:
    grad_cam = GradCAM(model)
    try:
        cnn_map = grad_cam.generate(image_tensor)
    finally:
        grad_cam.close()

    rollout = AttentionRollout(model)
    vit_map = rollout.generate(image_tensor)

    return ExplainabilityResult(
        cnn_heatmap_base64=_overlay_to_base64(cnn_map, image_bgr),
        vit_attention_base64=_overlay_to_base64(vit_map, image_bgr),
    )
