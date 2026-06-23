from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import cv2
import numpy as np

from .pipeline import Detection


@dataclass
class DecisionConfig:
    center_tolerance: float = 0.08
    near_zone_ratio: float = 0.68
    caution_zone_ratio: float = 0.52
    stop_area_ratio: float = 0.012
    lane_area_ratio: float = 0.002


@dataclass
class VehicleDecision:
    action: str
    steering_hint: str
    lateral_offset: float
    target_speed: str
    risk_level: str
    confidence: float
    reasons: list[str]
    signals: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _in_center_band(detection: Detection, width: int, band_ratio: float = 0.38) -> bool:
    cx, _ = detection.centroid
    band = width * band_ratio
    return (width / 2 - band) <= cx <= (width / 2 + band)


def _near_vehicle(detection: Detection, height: int, ratio: float) -> bool:
    _, y, _, h = detection.bbox
    return y + h >= height * ratio


def _estimate_lateral_offset(
    detections: Iterable[Detection],
    width: int,
    height: int,
    config: DecisionConfig,
) -> tuple[float, float, list[str]]:
    lane_candidates = [
        detection
        for detection in detections
        if detection.marking_type == "lane line"
        and detection.area >= width * height * config.lane_area_ratio
        and _near_vehicle(detection, height, config.caution_zone_ratio)
    ]
    if not lane_candidates:
        return 0.0, 0.35, ["Lane confidence low: no strong lane line found near vehicle."]

    left = [item for item in lane_candidates if item.centroid[0] < width / 2]
    right = [item for item in lane_candidates if item.centroid[0] >= width / 2]
    reasons: list[str] = []

    if left and right:
        left_x = max(item.centroid[0] for item in left)
        right_x = min(item.centroid[0] for item in right)
        lane_center = (left_x + right_x) / 2
        confidence = 0.88
        reasons.append("Left and right lane boundaries detected.")
    else:
        strongest = max(lane_candidates, key=lambda item: item.area)
        lane_center = strongest.centroid[0]
        confidence = 0.55
        side = "left" if strongest.centroid[0] < width / 2 else "right"
        reasons.append(f"Only {side} lane boundary detected; steering hint is approximate.")

    offset = (lane_center - width / 2) / max(1.0, width / 2)
    return round(float(offset), 3), confidence, reasons


def make_vehicle_decision(
    detections: list[Detection],
    frame_shape: tuple[int, int] | tuple[int, int, int],
    config: DecisionConfig | None = None,
) -> VehicleDecision:
    config = config or DecisionConfig()
    height, width = frame_shape[:2]
    frame_area = width * height
    offset, lane_confidence, reasons = _estimate_lateral_offset(
        detections,
        width,
        height,
        config,
    )

    signals: list[str] = []
    risk = "low"
    target_speed = "normal"
    action = "keep_lane"

    near_stop = [
        item
        for item in detections
        if item.marking_type == "stop/crosswalk stripe"
        and _near_vehicle(item, height, config.near_zone_ratio)
        and _in_center_band(item, width)
        and item.area >= frame_area * config.stop_area_ratio
    ]
    if near_stop:
        signals.append("stop_or_crosswalk_marking")
        reasons.append("Stop/crosswalk-style marking detected close to vehicle path.")
        action = "prepare_to_stop"
        target_speed = "slow"
        risk = "high"

    arrows = [
        item
        for item in detections
        if item.marking_type == "arrow or symbol"
        and _near_vehicle(item, height, config.caution_zone_ratio)
    ]
    if arrows:
        signals.append("direction_or_lane_symbol")
        reasons.append("Arrow/symbol road marking detected; route planner should confirm permitted movement.")
        if action == "keep_lane":
            action = "follow_marking"
            target_speed = "normal"
            risk = "medium"

    yellow_near = [
        item
        for item in detections
        if item.color == "yellow" and _near_vehicle(item, height, config.caution_zone_ratio)
    ]
    if yellow_near:
        signals.append("yellow_caution_marking")
        reasons.append("Yellow chromaticity marking detected near driving corridor.")
        if risk == "low":
            risk = "medium"

    red_near = [
        item
        for item in detections
        if item.color == "red" and _near_vehicle(item, height, config.caution_zone_ratio)
    ]
    if red_near:
        signals.append("red_restricted_or_warning_marking")
        reasons.append("Red chromaticity marking detected; treat as restricted/warning zone until map confirms.")
        action = "slow_and_verify"
        target_speed = "slow"
        risk = "high"

    if abs(offset) > config.center_tolerance and action == "keep_lane":
        action = "correct_lane_position"
    if offset > config.center_tolerance:
        steering = "steer_left"
        reasons.append("Estimated lane center is right of vehicle center.")
    elif offset < -config.center_tolerance:
        steering = "steer_right"
        reasons.append("Estimated lane center is left of vehicle center.")
    else:
        steering = "hold_center"

    if not detections:
        action = "proceed_with_caution"
        steering = "hold_center"
        target_speed = "slow"
        risk = "medium"
        reasons = ["No road paint markings detected in the configured ROI."]
        signals = ["no_marking_detected"]

    confidence = lane_confidence
    if risk == "high":
        confidence = min(0.95, confidence + 0.05)
    elif signals:
        confidence = min(0.9, confidence + 0.03)

    return VehicleDecision(
        action=action,
        steering_hint=steering,
        lateral_offset=offset,
        target_speed=target_speed,
        risk_level=risk,
        confidence=round(float(confidence), 3),
        reasons=reasons,
        signals=signals or ["lane_marking_context"],
    )


def draw_decision_hud(image_bgr: np.ndarray, decision: VehicleDecision) -> np.ndarray:
    output = image_bgr.copy()
    risk_color = {
        "low": (60, 210, 80),
        "medium": (0, 210, 255),
        "high": (45, 45, 230),
    }.get(decision.risk_level, (255, 255, 255))
    lines = [
        f"ACTION: {decision.action}",
        f"STEER: {decision.steering_hint} | SPEED: {decision.target_speed}",
        f"RISK: {decision.risk_level} | CONF: {decision.confidence:.2f}",
    ]
    x, y = 16, 28
    cv2.rectangle(output, (8, 8), (520, 108), (20, 20, 20), thickness=-1)
    cv2.rectangle(output, (8, 8), (520, 108), risk_color, thickness=2)
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (x, y + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            risk_color if index == 0 else (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

    height, width = output.shape[:2]
    center_x = width // 2
    target_x = int(center_x + decision.lateral_offset * (width / 2))
    cv2.line(output, (center_x, height), (center_x, int(height * 0.58)), (255, 255, 255), 2)
    cv2.arrowedLine(
        output,
        (center_x, height - 30),
        (target_x, height - 90),
        risk_color,
        4,
        tipLength=0.25,
    )
    return output
