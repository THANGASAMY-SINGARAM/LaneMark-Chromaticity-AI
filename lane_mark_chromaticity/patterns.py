from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass

from .pipeline import Detection
from .vehicle_decision import VehicleDecision


@dataclass
class FramePattern:
    frame: int
    pattern: str
    confidence: float
    evidence: list[str]
    recommended_action: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VideoPatternSummary:
    dominant_pattern: str
    confidence: float
    frames_analyzed: int
    pattern_counts: dict[str, int]
    color_counts: dict[str, int]
    recommended_action: str
    evidence: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class RoadPatternTracker:
    """Temporal road-marking pattern recognition for vehicle video input."""

    def __init__(self, window_size: int = 12) -> None:
        self.window_size = max(3, window_size)
        self.recent_patterns: deque[str] = deque(maxlen=self.window_size)
        self.pattern_counts: Counter[str] = Counter()
        self.color_counts: Counter[str] = Counter()
        self.frames_analyzed = 0

    def update(
        self,
        frame_index: int,
        detections: list[Detection],
        decision: VehicleDecision,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> FramePattern:
        height, width = frame_shape[:2]
        pattern, evidence, confidence = classify_frame_pattern(detections, height, width)
        action = action_for_pattern(pattern, decision)

        self.frames_analyzed += 1
        self.pattern_counts[pattern] += 1
        self.recent_patterns.append(pattern)
        for detection in detections:
            self.color_counts[detection.color] += 1

        return FramePattern(
            frame=frame_index,
            pattern=pattern,
            confidence=confidence,
            evidence=evidence,
            recommended_action=action,
        )

    def summary(self) -> VideoPatternSummary:
        if not self.frames_analyzed:
            return VideoPatternSummary(
                dominant_pattern="no_video_frames",
                confidence=0.0,
                frames_analyzed=0,
                pattern_counts={},
                color_counts={},
                recommended_action="check_video_input",
                evidence=["No frames were decoded from the input video."],
            )

        dominant, count = self.pattern_counts.most_common(1)[0]
        recent = Counter(self.recent_patterns)
        recent_dominant, recent_count = recent.most_common(1)[0]
        stable_recent = recent_dominant == dominant and recent_count >= max(2, len(self.recent_patterns) // 2)
        confidence = count / max(1, self.frames_analyzed)
        if stable_recent:
            confidence = min(1.0, confidence + 0.12)

        return VideoPatternSummary(
            dominant_pattern=dominant,
            confidence=round(confidence, 3),
            frames_analyzed=self.frames_analyzed,
            pattern_counts=dict(self.pattern_counts),
            color_counts=dict(self.color_counts),
            recommended_action=summary_action(dominant),
            evidence=[
                f"Dominant pattern appeared in {count}/{self.frames_analyzed} frames.",
                f"Recent temporal window: {dict(recent)}.",
            ],
        )


def classify_frame_pattern(
    detections: list[Detection],
    height: int,
    width: int,
) -> tuple[str, list[str], float]:
    if not detections:
        return "no_visible_marking", ["No chromatic road paint detected."], 0.35

    frame_area = height * width
    near = [item for item in detections if item.bbox[1] + item.bbox[3] > height * 0.58]
    lane_lines = [item for item in near if item.marking_type == "lane line"]
    stop_stripes = [item for item in near if item.marking_type == "stop/crosswalk stripe"]
    arrows = [item for item in near if item.marking_type == "arrow or symbol"]
    yellow = [item for item in near if item.color == "yellow"]
    red = [item for item in near if item.color == "red"]
    white = [item for item in near if item.color == "white"]

    evidence: list[str] = []

    if len(stop_stripes) >= 2:
        evidence.append(f"{len(stop_stripes)} repeated stop/crosswalk stripe components near vehicle.")
        return "crosswalk_or_stop_bar_pattern", evidence, 0.86

    if arrows:
        evidence.append(f"{len(arrows)} arrow/symbol component(s) detected in driving corridor.")
        return "direction_arrow_or_lane_symbol_pattern", evidence, 0.78

    if red and sum(item.area for item in red) > frame_area * 0.006:
        evidence.append("Large red chromaticity marking appears near vehicle path.")
        return "restricted_or_warning_color_pattern", evidence, 0.82

    if yellow and sum(item.area for item in yellow) > frame_area * 0.012:
        evidence.append("Large yellow chromaticity region appears near vehicle path.")
        return "yellow_caution_or_lane_boundary_pattern", evidence, 0.76

    if len(lane_lines) >= 2:
        left = any(item.centroid[0] < width / 2 for item in lane_lines)
        right = any(item.centroid[0] >= width / 2 for item in lane_lines)
        if left and right:
            evidence.append("Lane-line components appear on both sides of vehicle center.")
            return "centered_lane_corridor_pattern", evidence, 0.84

    if lane_lines:
        evidence.append("Lane-line pattern visible on one side of vehicle corridor.")
        return "single_lane_boundary_pattern", evidence, 0.68

    if len(white) >= 2:
        evidence.append(f"{len(white)} white road-marking components detected near vehicle.")
        return "white_road_marking_pattern", evidence, 0.62

    largest = max(detections, key=lambda item: item.area)
    evidence.append(f"Largest marking is {largest.color} {largest.marking_type}.")
    return "generic_road_marking_pattern", evidence, 0.55


def action_for_pattern(pattern: str, decision: VehicleDecision) -> str:
    if pattern == "crosswalk_or_stop_bar_pattern":
        return "slow_prepare_stop"
    if pattern == "restricted_or_warning_color_pattern":
        return "slow_verify_restriction"
    if pattern == "direction_arrow_or_lane_symbol_pattern":
        return "follow_route_after_symbol_confirmation"
    if pattern in {"centered_lane_corridor_pattern", "single_lane_boundary_pattern"}:
        return decision.action
    if pattern == "no_visible_marking":
        return "slow_use_other_sensors"
    return decision.action


def summary_action(pattern: str) -> str:
    return {
        "crosswalk_or_stop_bar_pattern": "vehicle_should_slow_and_prepare_to_stop",
        "restricted_or_warning_color_pattern": "vehicle_should_slow_and_verify_restriction",
        "direction_arrow_or_lane_symbol_pattern": "vehicle_should_confirm_route_direction",
        "yellow_caution_or_lane_boundary_pattern": "vehicle_should_follow_caution_boundary",
        "centered_lane_corridor_pattern": "vehicle_can_hold_lane",
        "single_lane_boundary_pattern": "vehicle_can_track_visible_boundary_with_caution",
        "no_visible_marking": "vehicle_should_slow_and_use_other_sensors",
    }.get(pattern, "vehicle_should_continue_with_caution")
