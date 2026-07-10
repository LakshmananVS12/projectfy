from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34


@dataclass
class HybridDetectorConfig:
    """Configuration for the hybrid CNN + ViT detector."""

    num_classes: int = 4
    class_names: Tuple[str, ...] = (
        "pothole",
        "linear_crack",
        "alligator_crack",
        "edge_break",
    )
    cnn_variant: str = "resnet34"
    cnn_output_stage: str = "layer3"  # layer3 keeps stride-16 features aligned with ViT patch grid.
    vit_variant: str = "deit_small_patch16_224"
    fusion_channels: int = 256
    head_channels: int = 256
    head_depth: int = 4
    pretrained_backbones: bool = True
    prior_prob: float = 0.01


class ResNetFeatureExtractor(nn.Module):
    """Pretrained CNN branch for local texture and edge cues.

    This branch uses ImageNet-pretrained weights by default. The detection head is trained
    from scratch on top of these features.
    """

    def __init__(self, variant: str = "resnet34", output_stage: str = "layer3", pretrained: bool = True) -> None:
        super().__init__()
        if variant not in {"resnet18", "resnet34"}:
            raise ValueError(f"Unsupported cnn variant: {variant}")
        if output_stage not in {"layer3", "layer4"}:
            raise ValueError("cnn_output_stage must be 'layer3' or 'layer4'.")

        if variant == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            self._stage_channels = {"layer3": 256, "layer4": 512}
        else:
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = resnet34(weights=weights)
            self._stage_channels = {"layer3": 256, "layer4": 512}

        self.variant = variant
        self.output_stage = output_stage

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    @property
    def out_channels(self) -> int:
        return self._stage_channels[self.output_stage]

    @property
    def stride(self) -> int:
        return 16 if self.output_stage == "layer3" else 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.output_stage == "layer3":
            return x
        x = self.layer4(x)
        return x


class FCOSHead(nn.Module):
    """From-scratch FCOS-style anchor-free detection head.

    Unlike the CNN/ViT backbones, all parameters in this head are newly initialized and
    trained from scratch for road-damage detection.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feat_channels: int = 256,
        depth: int = 4,
        prior_prob: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        cls_layers: List[nn.Module] = []
        reg_layers: List[nn.Module] = []
        for layer_idx in range(depth):
            cls_in = in_channels if layer_idx == 0 else feat_channels
            reg_in = in_channels if layer_idx == 0 else feat_channels
            cls_layers.extend(
                [
                    nn.Conv2d(cls_in, feat_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.GroupNorm(32, feat_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            reg_layers.extend(
                [
                    nn.Conv2d(reg_in, feat_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.GroupNorm(32, feat_channels),
                    nn.ReLU(inplace=True),
                ]
            )

        self.cls_tower = nn.Sequential(*cls_layers)
        self.reg_tower = nn.Sequential(*reg_layers)

        self.cls_logits = nn.Conv2d(feat_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_reg = nn.Conv2d(feat_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(feat_channels, 1, kernel_size=3, stride=1, padding=1)

        self._init_weights(prior_prob=prior_prob)

    def _init_weights(self, prior_prob: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # FCOS-style prior to reduce initial false positives.
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_feat = self.cls_tower(feat)
        reg_feat = self.reg_tower(feat)

        cls_logits = self.cls_logits(cls_feat)
        bbox_reg = F.relu(self.bbox_reg(reg_feat))
        centerness = self.centerness(reg_feat)
        return cls_logits, bbox_reg, centerness


class HybridRoadDamageDetector(nn.Module):
    """Two-branch hybrid detector used for both photo and video frame inference."""

    def __init__(self, config: HybridDetectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or HybridDetectorConfig()

        # Pretrained branch: CNN local-feature extractor.
        self.cnn_branch = ResNetFeatureExtractor(
            variant=self.config.cnn_variant,
            output_stage=self.config.cnn_output_stage,
            pretrained=self.config.pretrained_backbones,
        )

        # Pretrained branch: ViT global-context extractor.
        self.vit_branch = timm.create_model(
            self.config.vit_variant,
            pretrained=self.config.pretrained_backbones,
            num_classes=0,
            global_pool="",
        )
        vit_channels = self.vit_branch.num_features

        # From-scratch fusion block to merge CNN + ViT spatial maps.
        self.fusion_reduce = nn.Sequential(
            nn.Conv2d(self.cnn_branch.out_channels + vit_channels, self.config.fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.config.fusion_channels),
            nn.ReLU(inplace=True),
        )

        # From-scratch FCOS-style detection head.
        self.detection_head = FCOSHead(
            in_channels=self.config.fusion_channels,
            num_classes=self.config.num_classes,
            feat_channels=self.config.head_channels,
            depth=self.config.head_depth,
            prior_prob=self.config.prior_prob,
        )

    @property
    def stride(self) -> int:
        return self.cnn_branch.stride

    def _vit_tokens_to_map(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        vit_feats = self.vit_branch.forward_features(x)

        if vit_feats.ndim == 4:
            # Some timm models can emit spatial maps directly.
            if vit_feats.shape[-2:] != target_hw:
                vit_feats = F.interpolate(vit_feats, size=target_hw, mode="bilinear", align_corners=False)
            return vit_feats

        if vit_feats.ndim != 3:
            raise RuntimeError(f"Unexpected ViT features shape: {tuple(vit_feats.shape)}")

        bsz, token_count, channels = vit_feats.shape

        grid_h = x.shape[-2] // 16
        grid_w = x.shape[-1] // 16
        expected_tokens = grid_h * grid_w

        # Drop non-spatial prefix tokens (e.g., cls token) when present.
        if token_count > expected_tokens:
            vit_feats = vit_feats[:, token_count - expected_tokens :, :]
            token_count = expected_tokens

        if token_count != expected_tokens:
            side = int(math.sqrt(token_count))
            if side * side != token_count:
                raise RuntimeError(
                    "Cannot infer ViT patch grid from token count. "
                    f"token_count={token_count}, expected={expected_tokens}."
                )
            grid_h = side
            grid_w = side

        vit_map = vit_feats.transpose(1, 2).reshape(bsz, channels, grid_h, grid_w)

        if vit_map.shape[-2:] != target_hw:
            vit_map = F.interpolate(vit_map, size=target_hw, mode="bilinear", align_corners=False)
        return vit_map

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cnn_map = self.cnn_branch(x)
        vit_map = self._vit_tokens_to_map(x, target_hw=cnn_map.shape[-2:])

        fused = torch.cat([cnn_map, vit_map], dim=1)
        fused = self.fusion_reduce(fused)

        cls_logits, bbox_reg, centerness = self.detection_head(fused)
        return {
            "cls_logits": cls_logits,
            "bbox_reg": bbox_reg,
            "centerness": centerness,
            "feature_map": fused,
            "stride": torch.tensor(self.stride, device=x.device, dtype=torch.int64),
        }

    def freeze_backbones(self) -> None:
        for module in (self.cnn_branch, self.vit_branch):
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_vit_last_blocks(self, num_blocks: int = 2) -> None:
        if not hasattr(self.vit_branch, "blocks"):
            return
        blocks = list(self.vit_branch.blocks)
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(self.vit_branch, "norm"):
            for param in self.vit_branch.norm.parameters():
                param.requires_grad = True

    def unfreeze_cnn_last_stage(self) -> None:
        for param in self.cnn_branch.layer3.parameters():
            param.requires_grad = True
        if self.config.cnn_output_stage == "layer4":
            for param in self.cnn_branch.layer4.parameters():
                param.requires_grad = True


def build_hybrid_model(config: HybridDetectorConfig | None = None) -> HybridRoadDamageDetector:
    return HybridRoadDamageDetector(config=config)
