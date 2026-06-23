from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


COLOR_ORDER = ("white", "yellow", "red", "blue")

PALETTE_BGR = {
    "white": (245, 245, 245),
    "yellow": (0, 220, 255),
    "red": (45, 45, 230),
    "blue": (230, 130, 40),
}


@dataclass(frozen=True)
class ColorMaskStats:
    color: str
    pixels: int
    coverage_ratio: float


@dataclass(frozen=True)
class ColorReport:
    total_pixels: int
    detected_pixels: int
    coverage_ratio: float
    dominant_color: str
    colors: list[ColorMaskStats]


PROFILE_ADJUSTMENTS = {
    "sensitive": {
        "white_min_value": -22,
        "white_max_saturation": 22,
        "yellow_min_saturation": -18,
        "yellow_min_value": -18,
        "red_min_saturation": -18,
        "red_min_value": -18,
        "blue_min_saturation": -18,
        "blue_min_value": -18,
    },
    "balanced": {},
    "strict": {
        "white_min_value": 18,
        "white_max_saturation": -18,
        "yellow_min_saturation": 22,
        "yellow_min_value": 18,
        "red_min_saturation": 22,
        "red_min_value": 18,
        "blue_min_saturation": 22,
        "blue_min_value": 18,
    },
}


def normalize_enabled_colors(colors: Iterable[str] | None) -> tuple[str, ...]:
    if colors is None:
        return COLOR_ORDER
    selected = tuple(color for color in COLOR_ORDER if color in set(colors))
    return selected or COLOR_ORDER


def _setting(config: object, name: str, minimum: int = 0, maximum: int = 255) -> int:
    profile = getattr(config, "detector_profile", "balanced")
    base = int(getattr(config, name))
    adjusted = base + PROFILE_ADJUSTMENTS.get(profile, {}).get(name, 0)
    return int(np.clip(adjusted, minimum, maximum))


def _enhance_for_color(image_bgr: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return image_bgr
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _clean_mask(mask: np.ndarray, kernel_size: int, blur_size: int) -> np.ndarray:
    if blur_size > 1:
        blur_size = int(blur_size) | 1
        mask = cv2.medianBlur(mask, blur_size)
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def build_color_masks(
    image_bgr: np.ndarray,
    roi_mask: np.ndarray,
    config: object,
) -> dict[str, np.ndarray]:
    working = _enhance_for_color(image_bgr, bool(getattr(config, "use_clahe", False)))
    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    white_hsv = cv2.inRange(
        hsv,
        (0, 0, _setting(config, "white_min_value")),
        (179, _setting(config, "white_max_saturation"), 255),
    )
    white_lab = cv2.inRange(l_channel, max(120, _setting(config, "white_min_value") + 15), 255)
    white = cv2.bitwise_and(white_hsv, white_lab)

    yellow = cv2.inRange(
        hsv,
        (15, _setting(config, "yellow_min_saturation"), _setting(config, "yellow_min_value")),
        (42, 255, 255),
    )
    red_low = cv2.inRange(
        hsv,
        (0, _setting(config, "red_min_saturation"), _setting(config, "red_min_value")),
        (10, 255, 255),
    )
    red_high = cv2.inRange(
        hsv,
        (170, _setting(config, "red_min_saturation"), _setting(config, "red_min_value")),
        (179, 255, 255),
    )
    red = cv2.bitwise_or(red_low, red_high)
    blue = cv2.inRange(
        hsv,
        (90, _setting(config, "blue_min_saturation"), _setting(config, "blue_min_value")),
        (130, 255, 255),
    )

    raw_masks = {"white": white, "yellow": yellow, "red": red, "blue": blue}
    enabled_colors = normalize_enabled_colors(getattr(config, "enabled_colors", COLOR_ORDER))
    return {
        color: _clean_mask(
            cv2.bitwise_and(raw_masks[color], roi_mask),
            int(getattr(config, "morph_kernel", 5)),
            int(getattr(config, "mask_blur", 3)),
        )
        for color in enabled_colors
    }


def color_report(masks: dict[str, np.ndarray]) -> ColorReport:
    if not masks:
        return ColorReport(0, 0, 0.0, "none", [])

    first = next(iter(masks.values()))
    total_pixels = int(first.shape[0] * first.shape[1])
    stats = [
        ColorMaskStats(
            color=color,
            pixels=int(cv2.countNonZero(mask)),
            coverage_ratio=round(float(cv2.countNonZero(mask) / max(1, total_pixels)), 5),
        )
        for color, mask in masks.items()
    ]
    detected_pixels = int(sum(item.pixels for item in stats))
    dominant = max(stats, key=lambda item: item.pixels).color if detected_pixels else "none"
    return ColorReport(
        total_pixels=total_pixels,
        detected_pixels=detected_pixels,
        coverage_ratio=round(float(detected_pixels / max(1, total_pixels)), 5),
        dominant_color=dominant,
        colors=stats,
    )
