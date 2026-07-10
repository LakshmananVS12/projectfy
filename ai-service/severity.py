from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


BBox = Sequence[float]


@dataclass(frozen=True)
class SeverityResult:
    level: str
    score: float
    width_ratio: float
    area_ratio: float


# Type-specific multipliers capture that the same footprint can imply different urgency.
_DAMAGE_MULTIPLIER: Dict[str, float] = {
    "pothole": 1.10,
    "linear_crack": 0.90,
    "alligator_crack": 1.20,
    "edge_break": 1.00,
}

# Thresholds are in normalized severity-score units.
# Score computation uses:
#   0.65 * area_ratio + 0.35 * width_ratio, then multiplied by type multiplier.
# Where:
#   width_ratio = bbox_width / road_width_px
#   area_ratio  = bbox_area / (road_width_px^2)
#
# This follows the project rule of "bbox area vs road width + damage type" while
# keeping values interpretable and easy to calibrate.
_SEVERITY_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    # (low_to_medium_boundary, medium_to_high_boundary)
    "pothole": (0.015, 0.050),
    "linear_crack": (0.010, 0.035),
    "alligator_crack": (0.012, 0.040),
    "edge_break": (0.012, 0.040),
}


def _safe_bbox(bbox: BBox) -> Tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def estimate_severity(
    bbox: BBox,
    damage_type: str,
    image_width: int,
    road_width_px: float | None = None,
) -> SeverityResult:
    """Estimate severity with rule-based geometry + damage-type adjustment.

    Args:
        bbox: [x1, y1, x2, y2] in pixels.
        damage_type: one of project classes.
        image_width: full image width in pixels.
        road_width_px: optional effective road width in pixels. If missing,
            defaults to 85% of image width as a practical lane-view heuristic.
    """

    x1, y1, x2, y2 = _safe_bbox(bbox)
    bbox_w = max(1.0, x2 - x1)
    bbox_h = max(1.0, y2 - y1)
    bbox_area = bbox_w * bbox_h

    if road_width_px is None:
        road_width_px = max(1.0, image_width * 0.85)
    else:
        road_width_px = max(1.0, float(road_width_px))

    width_ratio = bbox_w / road_width_px
    area_ratio = bbox_area / (road_width_px * road_width_px)

    base_score = 0.65 * area_ratio + 0.35 * width_ratio
    multiplier = _DAMAGE_MULTIPLIER.get(damage_type, 1.0)
    score = base_score * multiplier

    low_med, med_high = _SEVERITY_THRESHOLDS.get(damage_type, (0.012, 0.040))
    if score < low_med:
        level = "LOW"
    elif score < med_high:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return SeverityResult(
        level=level,
        score=float(score),
        width_ratio=float(width_ratio),
        area_ratio=float(area_ratio),
    )


def estimate_severity_for_detections(
    detections: Iterable[Dict[str, object]],
    image_width: int,
    road_width_px: float | None = None,
) -> List[Dict[str, object]]:
    """Attach severity fields in-place-friendly format for endpoint responses."""

    enriched: List[Dict[str, object]] = []
    for det in detections:
        cls_name = str(det.get("class", "unknown"))
        bbox = det.get("bbox")
        if not isinstance(bbox, (list, tuple)):
            enriched.append(dict(det))
            continue

        sev = estimate_severity(
            bbox=bbox,
            damage_type=cls_name,
            image_width=image_width,
            road_width_px=road_width_px,
        )
        row = dict(det)
        row["severity"] = sev.level
        row["severity_score"] = sev.score
        row["severity_debug"] = {
            "width_ratio": sev.width_ratio,
            "area_ratio": sev.area_ratio,
        }
        enriched.append(row)

    return enriched
