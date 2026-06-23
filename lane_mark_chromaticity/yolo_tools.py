from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml


CEYMO_CLASSES = [
    "BL",
    "CL",
    "DM",
    "JB",
    "LA",
    "PC",
    "RA",
    "SA",
    "SL",
    "SLA",
    "SRA",
]


def require_ultralytics():
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - depends on local native runtime.
        raise RuntimeError(
            "YOLOv8 is installed but could not be imported. This usually means "
            "PyTorch native DLL dependencies are missing. Reinstall torch/torchvision "
            "for your active Python, then retry."
        ) from exc
    return YOLO


def _normalise_polygon(points: list[list[float]], width: int, height: int) -> list[float]:
    values: list[float] = []
    for x, y in points:
        values.extend(
            [
                min(1.0, max(0.0, float(x) / width)),
                min(1.0, max(0.0, float(y) / height)),
            ]
        )
    return values


def convert_ceymo_to_yolo(
    source_root: str | Path,
    output_root: str | Path,
    split: str = "test",
    copy_images: bool = False,
) -> Path:
    source_root = Path(source_root)
    output_root = Path(output_root)
    image_dir = source_root / split / "images"
    annotation_dir = source_root / split / "polygon_annotations"

    if not image_dir.exists() or not annotation_dir.exists():
        raise FileNotFoundError(
            f"Expected {image_dir} and {annotation_dir}. Check source_root and split."
        )

    label_output = output_root / "labels" / split
    image_output = output_root / "images" / split
    label_output.mkdir(parents=True, exist_ok=True)
    image_output.mkdir(parents=True, exist_ok=True)

    class_to_id = {name: index for index, name in enumerate(CEYMO_CLASSES)}
    converted = 0

    for annotation_path in sorted(annotation_dir.glob("*.json")):
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        width = int(data.get("imageWidth", 0))
        height = int(data.get("imageHeight", 0))
        if width <= 0 or height <= 0:
            continue

        lines: list[str] = []
        for shape in data.get("shapes", []):
            label = shape.get("label")
            points = shape.get("points") or []
            if label not in class_to_id or len(points) < 3:
                continue
            polygon = _normalise_polygon(points, width, height)
            values = " ".join(f"{value:.6f}" for value in polygon)
            lines.append(f"{class_to_id[label]} {values}")

        (label_output / f"{annotation_path.stem}.txt").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )
        converted += 1

        if copy_images:
            image_path = image_dir / data.get("imagePath", f"{annotation_path.stem}.jpg")
            if image_path.exists():
                shutil.copy2(image_path, image_output / image_path.name)

    yaml_path = output_root / "ceymo-seg.yaml"
    dataset_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": f"images/{split}",
        "test": f"images/{split}",
        "names": {index: name for index, name in enumerate(CEYMO_CLASSES)},
    }
    yaml_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    return yaml_path


def train_yolo(data_yaml: str | Path, model: str, epochs: int, image_size: int):
    YOLO = require_ultralytics()
    yolo = YOLO(model)
    return yolo.train(data=str(data_yaml), epochs=epochs, imgsz=image_size)


def predict_yolo(weights: str | Path, source: str | Path, output_dir: str | Path):
    YOLO = require_ultralytics()
    yolo = YOLO(str(weights))
    return yolo.predict(source=str(source), project=str(output_dir), save=True)
