from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

TARGET_CLASSES: Tuple[str, ...] = (
    "pothole",
    "linear_crack",
    "alligator_crack",
    "edge_break",
)

# Canonical mapping from common RDD class labels to project targets.
# Notes:
# - D00 + D10 are merged into linear_crack.
# - D01 / D11 are treated as edge-break-like classes when present.
# - ravelling is optional and only included when sample counts are sufficient.
RAW_TO_TARGET: Dict[str, str] = {
    "d00": "linear_crack",
    "d10": "linear_crack",
    "d20": "alligator_crack",
    "d40": "pothole",
    "d01": "edge_break",
    "d11": "edge_break",
    "edge_break": "edge_break",
    "alligator_crack": "alligator_crack",
    "linear_crack": "linear_crack",
    "pothole": "pothole",
    "ravelling": "ravelling",
    "raveling": "ravelling",
    "d50": "ravelling",
}


@dataclass
class ClassDecision:
    active_classes: Tuple[str, ...]
    dropped_classes: Tuple[str, ...]
    notes: List[str]


def normalize_raw_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def map_raw_label(label: str) -> str | None:
    return RAW_TO_TARGET.get(normalize_raw_label(label))


def decide_active_classes(
    mapped_labels: Iterable[str],
    min_ravelling_samples: int = 300,
) -> ClassDecision:
    counts: Dict[str, int] = {}
    for label in mapped_labels:
        counts[label] = counts.get(label, 0) + 1

    notes: List[str] = []
    active = list(TARGET_CLASSES)
    dropped: List[str] = []

    ravelling_count = counts.get("ravelling", 0)
    if ravelling_count < min_ravelling_samples:
        dropped.append("ravelling")
        notes.append(
            "Ravelling samples are below viability threshold "
            f"({ravelling_count} < {min_ravelling_samples}); class dropped to avoid synthetic balancing."
        )
    else:
        active.append("ravelling")
        notes.append(
            f"Ravelling retained with {ravelling_count} samples (>= {min_ravelling_samples})."
        )

    edge_break_count = counts.get("edge_break", 0)
    if edge_break_count == 0:
        notes.append(
            "No edge_break-like labels were found in this raw dataset snapshot. "
            "Training can proceed with a reserved edge_break class index, but model quality for that class will require supplemental annotations."
        )

    return ClassDecision(
        active_classes=tuple(active),
        dropped_classes=tuple(dropped),
        notes=notes,
    )


def class_to_index(active_classes: Iterable[str]) -> Dict[str, int]:
    return {cls_name: idx for idx, cls_name in enumerate(active_classes)}
