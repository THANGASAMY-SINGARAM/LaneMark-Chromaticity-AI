from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .color_detection import COLOR_ORDER, PALETTE_BGR, build_color_masks, color_report


@dataclass
class DetectionConfig:
    resize_width: int = 1280
    roi_top_ratio: float = 0.58
    min_area: int = 400
    morph_kernel: int = 5
    mask_blur: int = 3
    detector_profile: str = "balanced"
    enabled_colors: tuple[str, ...] = COLOR_ORDER
    use_clahe: bool = False
    white_min_value: int = 150
    white_max_saturation: int = 90
    yellow_min_saturation: int = 55
    yellow_min_value: int = 90
    red_min_saturation: int = 60
    red_min_value: int = 70
    blue_min_saturation: int = 60
    blue_min_value: int = 60


@dataclass
class Detection:
    id: int
    color: str
    marking_type: str
    confidence: float
    area: float
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]
    chromaticity_rgb: tuple[float, float, float]
    mean_hsv: tuple[float, float, float]
    mask_coverage: float
    contour: list[list[int]]


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def resize_for_processing(image: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    if width <= 0 or image.shape[1] <= width:
        return image.copy(), 1.0
    scale = width / image.shape[1]
    height = int(round(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), scale


def build_roi_mask(shape: tuple[int, int], top_ratio: float) -> np.ndarray:
    height, width = shape
    top = int(np.clip(top_ratio, 0.0, 0.95) * height)
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(
        [
            [0, height - 1],
            [width - 1, height - 1],
            [int(width * 0.78), top],
            [int(width * 0.22), top],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def _clean_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def chromaticity_masks(image_bgr: np.ndarray, config: DetectionConfig) -> dict[str, np.ndarray]:
    roi = build_roi_mask(image_bgr.shape[:2], config.roi_top_ratio)
    return build_color_masks(image_bgr, roi, config)


def classify_shape(contour: np.ndarray) -> str:
    area = cv2.contourArea(contour)
    if area <= 0:
        return "marking"

    x, y, w, h = cv2.boundingRect(contour)
    aspect = max(w, h) / max(1, min(w, h))
    extent = area / max(1, w * h)
    hull = cv2.convexHull(contour)
    solidity = area / max(1.0, cv2.contourArea(hull))
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)

    if aspect >= 4.5 and extent > 0.28:
        return "lane line"
    if aspect >= 2.2 and extent > 0.45:
        return "stop/crosswalk stripe"
    if len(approx) >= 7 and solidity < 0.82:
        return "arrow or symbol"
    if 0.65 <= w / max(1, h) <= 1.55 and len(approx) <= 6:
        return "diamond/box marking"
    return "road marking"


def _mean_chromaticity(image_bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    pixels = image_bgr[mask > 0].astype(np.float32)
    if pixels.size == 0:
        return (0.0, 0.0, 0.0)
    rgb = pixels[:, ::-1]
    total = np.maximum(rgb.sum(axis=1, keepdims=True), 1.0)
    chroma = (rgb / total).mean(axis=0)
    return tuple(round(float(v), 4) for v in chroma)


def _mean_hsv(image_bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    pixels = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[mask > 0].astype(np.float32)
    if pixels.size == 0:
        return (0.0, 0.0, 0.0)
    return tuple(round(float(v), 2) for v in pixels.mean(axis=0))


def _confidence(area: float, mask_area: int, contour: np.ndarray) -> float:
    _, _, w, h = cv2.boundingRect(contour)
    extent = area / max(1, w * h)
    area_score = min(1.0, area / max(1000.0, mask_area * 0.02))
    extent_score = np.clip(extent * 1.4, 0.0, 1.0)
    return round(float(0.45 + 0.35 * area_score + 0.20 * extent_score), 3)


def extract_detections(
    image_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    config: DetectionConfig,
) -> list[Detection]:
    detections: list[Detection] = []
    next_id = 1

    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_area = int(cv2.countNonZero(mask))
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < config.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            component_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)
            moments = cv2.moments(contour)
            if moments["m00"]:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            detections.append(
                Detection(
                    id=next_id,
                    color=color,
                    marking_type=classify_shape(contour),
                    confidence=_confidence(area, mask_area, contour),
                    area=round(area, 2),
                    bbox=(int(x), int(y), int(w), int(h)),
                    centroid=(cx, cy),
                    chromaticity_rgb=_mean_chromaticity(image_bgr, component_mask),
                    mean_hsv=_mean_hsv(image_bgr, component_mask),
                    mask_coverage=round(float(area / max(1, mask.size)), 5),
                    contour=contour.reshape(-1, 2).astype(int).tolist(),
                )
            )
            next_id += 1

    return sorted(detections, key=lambda item: item.area, reverse=True)


def draw_overlay(
    image_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    detections: Iterable[Detection],
    alpha: float = 0.42,
) -> np.ndarray:
    overlay = image_bgr.copy()
    color_layer = np.zeros_like(image_bgr)

    for color, mask in masks.items():
        color_layer[mask > 0] = PALETTE_BGR[color]

    overlay = cv2.addWeighted(overlay, 1.0, color_layer, alpha, 0)
    for detection in detections:
        x, y, w, h = detection.bbox
        color = PALETTE_BGR.get(detection.color, (255, 255, 255))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        label = f"{detection.id}: {detection.color} {detection.marking_type}"
        cv2.putText(
            overlay,
            label,
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def analyze_image(
    image_bgr: np.ndarray,
    config: DetectionConfig | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[Detection]]:
    config = config or DetectionConfig()
    resized, _ = resize_for_processing(image_bgr, config.resize_width)
    masks = chromaticity_masks(resized, config)
    detections = extract_detections(resized, masks, config)
    overlay = draw_overlay(resized, masks, detections)
    return overlay, masks, detections


def combined_mask(masks: dict[str, np.ndarray]) -> np.ndarray:
    if not masks:
        return np.zeros((1, 1), dtype=np.uint8)
    result = np.zeros(next(iter(masks.values())).shape, dtype=np.uint8)
    for mask in masks.values():
        result = cv2.bitwise_or(result, mask)
    return result


def detections_to_records(detections: Iterable[Detection]) -> list[dict]:
    return [asdict(detection) for detection in detections]


def masks_to_records(masks: dict[str, np.ndarray]) -> list[dict]:
    report = color_report(masks)
    return [asdict(item) for item in report.colors]


def save_reports(
    detections: Iterable[Detection],
    output_json: str | Path,
    output_csv: str | Path | None = None,
) -> None:
    records = detections_to_records(detections)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(records, indent=2), encoding="utf-8")

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "id",
            "color",
            "marking_type",
            "confidence",
            "area",
            "bbox",
            "centroid",
            "chromaticity_rgb",
            "mean_hsv",
            "mask_coverage",
        ]
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record in records:
                writer.writerow({column: record[column] for column in columns})


def process_image_file(
    image_path: str | Path,
    output_dir: str | Path,
    config: DetectionConfig | None = None,
) -> dict[str, Path | int]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    overlay, masks, detections = analyze_image(image, config)

    stem = image_path.stem
    overlay_path = output_dir / f"{stem}_overlay.jpg"
    mask_path = output_dir / f"{stem}_mask.png"
    json_path = output_dir / f"{stem}_detections.json"
    csv_path = output_dir / f"{stem}_detections.csv"

    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(mask_path), combined_mask(masks))
    save_reports(detections, json_path, csv_path)

    return {
        "detections": len(detections),
        "overlay": overlay_path,
        "mask": mask_path,
        "json": json_path,
        "csv": csv_path,
    }
