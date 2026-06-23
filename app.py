from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import cv2
import imageio.v2 as imageio
import pandas as pd
import streamlit as st

from lane_mark_chromaticity.color_detection import COLOR_ORDER, PALETTE_BGR, color_report
from lane_mark_chromaticity.patterns import RoadPatternTracker
from lane_mark_chromaticity.pipeline import (
    DetectionConfig,
    analyze_image,
    combined_mask,
    detections_to_records,
    load_image,
    masks_to_records,
    process_image_file,
)
from lane_mark_chromaticity.vehicle_decision import draw_decision_hud, make_vehicle_decision


def bgr_to_rgb(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def fit_width(frame_bgr, width):
    if frame_bgr.shape[1] <= width:
        return frame_bgr
    height = int(frame_bgr.shape[0] * (width / frame_bgr.shape[1]))
    return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)


def chromaticity_view(frame_bgr, masks):
    color_layer = frame_bgr.copy()
    chroma = frame_bgr.copy()
    for color, mask in masks.items():
        chroma[mask > 0] = PALETTE_BGR[color]
    return cv2.addWeighted(color_layer, 0.35, chroma, 0.65, 0)


def detection_table(detections):
    records = detections_to_records(detections)
    return pd.DataFrame(
        [
            {
                "id": item["id"],
                "color": item["color"],
                "type": item["marking_type"],
                "confidence": item["confidence"],
                "area": item["area"],
                "coverage": item["mask_coverage"],
                "bbox": item["bbox"],
                "chromaticity_rgb": item["chromaticity_rgb"],
            }
            for item in records
        ]
    )


st.set_page_config(page_title="LaneMark Chromaticity AI", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 12px 14px;
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {gap: 0.7rem;}
    h1, h2, h3 {letter-spacing: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("LaneMark Chromaticity AI")

with st.sidebar:
    st.header("Input")
    input_mode = st.radio("Mode", ["Image", "Video"], horizontal=True)
    upload = st.file_uploader(
        "Road image" if input_mode == "Image" else "Road video",
        type=["jpg", "jpeg", "png", "bmp"] if input_mode == "Image" else ["mp4", "avi", "mov"],
    )
    sample_dir = Path("test/images")
    sample_images = sorted(sample_dir.glob("*.jpg"))[:120] if sample_dir.exists() else []
    sample = st.selectbox("CeyMo sample", [""] + [str(path) for path in sample_images])

    st.header("Detector")
    detector_profile = st.radio(
        "Profile",
        ["sensitive", "balanced", "strict"],
        horizontal=True,
        index=1,
    )
    enabled_colors = tuple(
        color
        for color in COLOR_ORDER
        if st.checkbox(color.title(), value=True, key=f"color_{color}")
    )
    use_clahe = st.toggle("Contrast equalization", value=False)

    st.header("Geometry")
    resize_width = st.slider("Resize width", 640, 1920, 1280, 80)
    roi_top_ratio = st.slider("ROI top", 0.05, 0.75, 0.58, 0.01)
    min_area = st.slider("Minimum area", 50, 6000, 400, 50)
    morph_kernel = st.slider("Cleanup kernel", 3, 17, 5, 2)
    mask_blur = st.slider("Mask blur", 1, 11, 3, 2)

config = DetectionConfig(
    resize_width=resize_width,
    roi_top_ratio=roi_top_ratio,
    min_area=min_area,
    morph_kernel=morph_kernel,
    mask_blur=mask_blur,
    detector_profile=detector_profile,
    enabled_colors=enabled_colors,
    use_clahe=use_clahe,
)

source_path = None
image_bgr = None

if upload is not None and input_mode == "Image":
    suffix = Path(upload.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(upload.getbuffer())
        source_path = Path(handle.name)
    image_bgr = load_image(source_path)
elif sample and input_mode == "Image":
    source_path = Path(sample)
    image_bgr = load_image(source_path)

if upload is not None and input_mode == "Video":
    suffix = Path(upload.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(upload.getbuffer())
        video_path = Path(handle.name)

    probe = cv2.VideoCapture(str(video_path))
    source_fps = probe.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_duration = total_frames / source_fps if total_frames else 0
    probe.release()

    m1, m2, m3 = st.columns(3)
    m1.metric("Duration", f"{source_duration:.1f}s")
    m2.metric("FPS", f"{source_fps:.1f}")
    m3.metric("Frames", f"{total_frames}")

    if st.button("Analyze video", type="primary"):
        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or source_fps or 25
        rows = []
        discovered_colors = Counter()
        discovered_markings = Counter()
        output_dir = Path("outputs") / "streamlit_video"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video = output_dir / "side_by_side_chromaticity.mp4"
        output_gif = output_dir / "side_by_side_chromaticity_preview.gif"
        output_json = output_dir / "video_patterns.json"
        writer = None
        tracker = RoadPatternTracker()
        frame_index = 0
        gif_frames = []
        preview_fps = min(8, max(1, int(round(fps))))
        gif_stride = max(1, int(round(fps / preview_fps)))
        progress = st.progress(0)
        status = st.empty()

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            overlay, masks, detections = analyze_image(frame, config)
            decision = make_vehicle_decision(detections, overlay.shape)
            frame_pattern = tracker.update(frame_index, detections, decision, overlay.shape)
            for detection in detections:
                discovered_colors[detection.color] += 1
                discovered_markings[f"{detection.color} {detection.marking_type}"] += 1
            marked_frame = draw_decision_hud(overlay, decision)
            cv2.putText(
                marked_frame,
                f"PATTERN: {frame_pattern.pattern}",
                (16, 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            side_by_side = cv2.hconcat([marked_frame, chromaticity_view(overlay, masks)])
            if writer is None:
                height, width = side_by_side.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            writer.write(side_by_side)
            if frame_index % gif_stride == 0:
                preview_frame = fit_width(side_by_side, 1100)
                gif_frames.append(bgr_to_rgb(preview_frame))
            rows.append(
                {
                    "frame": frame_index,
                    "pattern": frame_pattern.pattern,
                    "pattern_action": frame_pattern.recommended_action,
                    "vehicle_action": decision.action,
                    "steering": decision.steering_hint,
                    "speed": decision.target_speed,
                    "risk": decision.risk_level,
                    "confidence": decision.confidence,
                    "detections": len(detections),
                }
            )
            frame_index += 1
            if total_frames:
                progress.progress(min(1.0, frame_index / total_frames))
            if frame_index % 15 == 0:
                status.write(f"Analyzed {frame_index}/{total_frames or '?'} frames")

        capture.release()
        if writer is not None:
            writer.release()
        progress.progress(1.0)
        status.write(f"Analyzed {frame_index} frames")

        summary = tracker.summary().to_dict()
        output_json.write_text(pd.DataFrame(rows).to_json(orient="records", indent=2), encoding="utf-8")

        preview_tab, summary_tab, frames_tab = st.tabs(["Preview", "Summary", "Frame Log"])
        with preview_tab:
            if gif_frames:
                imageio.mimsave(output_gif, gif_frames, duration=gif_stride / fps)
                st.image(output_gif.read_bytes(), use_column_width=True)
            elif output_video.exists() and output_video.stat().st_size > 0:
                st.video(output_video.read_bytes())
        with summary_tab:
            c1, c2, c3 = st.columns(3)
            c1.metric("Dominant Pattern", summary["dominant_pattern"])
            c2.metric("Pattern Confidence", summary["confidence"])
            c3.metric("Recommended Action", summary["recommended_action"])
            st.dataframe(
                pd.DataFrame(
                    discovered_markings.most_common(),
                    columns=["marking", "count"],
                ),
                use_container_width=True,
                hide_index=True,
            )
        with frames_tab:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.stop()

if image_bgr is None:
    st.info("Upload a road image or choose a sample from the sidebar.")
    st.stop()

overlay, masks, detections = analyze_image(image_bgr, config)
decision = make_vehicle_decision(detections, overlay.shape)
decision_overlay = draw_decision_hud(overlay, decision)
mask = combined_mask(masks)
report = color_report(masks)

metric_cols = st.columns(5)
metric_cols[0].metric("Detections", len(detections))
metric_cols[1].metric("Dominant Color", report.dominant_color.title())
metric_cols[2].metric("Color Coverage", f"{report.coverage_ratio * 100:.2f}%")
metric_cols[3].metric("Vehicle Action", decision.action)
metric_cols[4].metric("Risk", decision.risk_level)

preview_tab, detection_tab, color_tab, decision_tab, export_tab = st.tabs(
    ["Preview", "Detections", "Colors", "Decision", "Export"]
)

with preview_tab:
    left, right = st.columns([1.2, 0.8])
    with left:
        st.image(bgr_to_rgb(decision_overlay), use_column_width=True)
    with right:
        st.image(mask, clamp=True, use_column_width=True)
        st.image(bgr_to_rgb(chromaticity_view(overlay, masks)), use_column_width=True)

with detection_tab:
    table = detection_table(detections)
    if table.empty:
        st.warning("No markings found. Try a sensitive profile, lower minimum area, or move the ROI higher.")
    else:
        st.dataframe(table, use_container_width=True, hide_index=True)

with color_tab:
    color_rows = pd.DataFrame(masks_to_records(masks))
    if color_rows.empty:
        st.warning("No enabled colors were detected.")
    else:
        color_rows["coverage_percent"] = (color_rows["coverage_ratio"] * 100).round(3)
        st.dataframe(color_rows, use_container_width=True, hide_index=True)
        st.bar_chart(color_rows.set_index("color")["pixels"])

with decision_tab:
    st.json(decision.to_dict())

with export_tab:
    if source_path is None:
        st.warning("Export is available after an upload or sample selection.")
    elif st.button("Export overlay, mask, JSON, and CSV", type="primary"):
        result = process_image_file(source_path, Path("outputs"), config)
        st.success(f"Exported {result['detections']} detections to {Path('outputs').resolve()}")
