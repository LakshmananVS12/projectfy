from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Sequence, Tuple
import shutil
import xml.etree.ElementTree as ET

import cv2

from class_map import class_to_index, decide_active_classes, map_raw_label


@dataclass
class ObjectAnnotation:
    class_name: str
    bbox: Tuple[float, float, float, float]


@dataclass
class ImageRecord:
    image_path: Path
    width: int
    height: int
    annotations: List[ObjectAnnotation]


def _collect_images(raw_dir: Path) -> Dict[str, Path]:
    image_map: Dict[str, Path] = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for p in raw_dir.rglob(ext):
            image_map[p.name.lower()] = p
            image_map[p.stem.lower()] = p
    return image_map


def _parse_xml(xml_file: Path, image_map: Dict[str, Path]) -> ImageRecord | None:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = (root.findtext("filename") or "").strip()
    key = filename.lower() if filename else xml_file.stem.lower()

    image_path = image_map.get(key)
    if image_path is None:
        image_path = image_map.get(Path(filename).stem.lower())
    if image_path is None:
        return None

    size = root.find("size")
    width = int(size.findtext("width")) if size is not None and size.find("width") is not None else 0
    height = int(size.findtext("height")) if size is not None and size.find("height") is not None else 0

    annotations: List[ObjectAnnotation] = []
    for obj in root.findall("object"):
        raw_label = obj.findtext("name")
        if not raw_label:
            continue
        mapped = map_raw_label(raw_label)
        if mapped is None:
            continue

        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        x1 = float(bnd.findtext("xmin", default="0"))
        y1 = float(bnd.findtext("ymin", default="0"))
        x2 = float(bnd.findtext("xmax", default="0"))
        y2 = float(bnd.findtext("ymax", default="0"))

        if x2 <= x1 or y2 <= y1:
            continue

        annotations.append(ObjectAnnotation(class_name=mapped, bbox=(x1, y1, x2, y2)))

    if not annotations:
        return None

    if width <= 0 or height <= 0:
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        height, width = image.shape[:2]

    return ImageRecord(image_path=image_path, width=width, height=height, annotations=annotations)


def _primary_class(record: ImageRecord) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for ann in record.annotations:
        counts[ann.class_name] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _stratified_split(
    records: Sequence[ImageRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[ImageRecord], List[ImageRecord], List[ImageRecord]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    groups: Dict[str, List[ImageRecord]] = defaultdict(list)
    for rec in records:
        groups[_primary_class(rec)].append(rec)

    rng = Random(seed)
    train: List[ImageRecord] = []
    val: List[ImageRecord] = []
    test: List[ImageRecord] = []

    for _, items in groups.items():
        local = list(items)
        rng.shuffle(local)

        n = len(local)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 5:
            if n_test == 0:
                n_test = 1
            if n_val == 0:
                n_val = 1
        if n_test + n_val >= n:
            n_test = max(0, n - 2)
            n_val = 1 if n >= 2 else 0

        n_train = n - n_val - n_test
        train.extend(local[:n_train])
        val.extend(local[n_train : n_train + n_val])
        test.extend(local[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _resize_and_scale_boxes(
    image,
    annotations: Sequence[ObjectAnnotation],
    out_size: int,
) -> Tuple[any, List[ObjectAnnotation]]:
    src_h, src_w = image.shape[:2]
    resized = cv2.resize(image, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    sx = out_size / float(src_w)
    sy = out_size / float(src_h)

    scaled: List[ObjectAnnotation] = []
    for ann in annotations:
        x1, y1, x2, y2 = ann.bbox
        scaled.append(
            ObjectAnnotation(
                class_name=ann.class_name,
                bbox=(x1 * sx, y1 * sy, x2 * sx, y2 * sy),
            )
        )

    return resized, scaled


def _export_split(
    split_name: str,
    split_records: Sequence[ImageRecord],
    out_dir: Path,
    out_size: int,
    class_to_idx: Dict[str, int],
) -> Dict[str, object]:
    images_dir = out_dir / "images" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": idx, "name": name}
            for name, idx in sorted(class_to_idx.items(), key=lambda kv: kv[1])
        ],
    }

    ann_id = 1
    for img_id, rec in enumerate(split_records, start=1):
        image = cv2.imread(str(rec.image_path))
        if image is None:
            continue

        processed, scaled_anns = _resize_and_scale_boxes(image, rec.annotations, out_size)
        out_name = f"{img_id:07d}_{rec.image_path.name}"
        out_path = images_dir / out_name
        cv2.imwrite(str(out_path), processed)

        coco["images"].append(
            {
                "id": img_id,
                "file_name": out_name,
                "width": out_size,
                "height": out_size,
            }
        )

        for ann in scaled_anns:
            if ann.class_name not in class_to_idx:
                continue
            x1, y1, x2, y2 = ann.bbox
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_to_idx[ann.class_name],
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    ann_path = out_dir / f"annotations_{split_name}.json"
    ann_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    return {
        "split": split_name,
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "annotation_file": str(ann_path),
    }


def run_preprocessing(
    raw_dir: Path,
    output_dir: Path,
    image_size: int,
    seed: int,
    min_ravelling_samples: int,
) -> None:
    if not raw_dir.exists():
        raise FileNotFoundError(
            "RDD2022 raw directory not found. Place the downloaded dataset under the provided path "
            "and rerun this script."
        )

    image_map = _collect_images(raw_dir)
    xml_files = list(raw_dir.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError("No XML annotations found. Expected Pascal VOC-style labels from RDD datasets.")

    records: List[ImageRecord] = []
    mapped_labels: List[str] = []

    for xml_file in xml_files:
        rec = _parse_xml(xml_file, image_map)
        if rec is None:
            continue
        records.append(rec)
        mapped_labels.extend(ann.class_name for ann in rec.annotations)

    if not records:
        raise RuntimeError("No usable annotated records found after mapping classes.")

    class_decision = decide_active_classes(
        mapped_labels=mapped_labels,
        min_ravelling_samples=min_ravelling_samples,
    )
    class_to_idx = class_to_index(class_decision.active_classes)

    # Remove dropped classes from image annotations.
    filtered: List[ImageRecord] = []
    dropped_set = set(class_decision.dropped_classes)
    for rec in records:
        anns = [ann for ann in rec.annotations if ann.class_name not in dropped_set and ann.class_name in class_to_idx]
        if anns:
            filtered.append(
                ImageRecord(
                    image_path=rec.image_path,
                    width=rec.width,
                    height=rec.height,
                    annotations=anns,
                )
            )

    train, val, test = _stratified_split(filtered, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_summaries = [
        _export_split("train", train, output_dir, image_size, class_to_idx),
        _export_split("val", val, output_dir, image_size, class_to_idx),
        _export_split("test", test, output_dir, image_size, class_to_idx),
    ]

    final_counts: Dict[str, int] = defaultdict(int)
    for rec in filtered:
        for ann in rec.annotations:
            final_counts[ann.class_name] += 1

    report = {
        "raw_records": len(records),
        "filtered_records": len(filtered),
        "active_classes": list(class_decision.active_classes),
        "dropped_classes": list(class_decision.dropped_classes),
        "class_counts": dict(sorted(final_counts.items())),
        "splits": split_summaries,
        "notes": class_decision.notes,
    }

    report_path = output_dir / "preprocess_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess RDD2022 into train/val/test splits for hybrid detection.")
    parser.add_argument("--raw-dir", required=True, type=Path, help="Path to raw downloaded RDD2022 data.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output path for processed dataset.")
    parser.add_argument("--image-size", default=640, type=int, help="Square resize for training images.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for deterministic split.")
    parser.add_argument(
        "--min-ravelling-samples",
        default=300,
        type=int,
        help="Minimum sample count required to keep ravelling as an active class.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_preprocessing(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        seed=args.seed,
        min_ravelling_samples=args.min_ravelling_samples,
    )


if __name__ == "__main__":
    main()
