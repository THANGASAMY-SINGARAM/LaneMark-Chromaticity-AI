# LaneMark Chromaticity AI

LaneMark Chromaticity AI detects road paint markings from images, videos, or a webcam using color/chromaticity segmentation and contour analysis. It highlights white, yellow, red, and blue road markings, estimates the marking shape, and exports overlays plus JSON/CSV reports.

The project is designed as a vehicle perception prototype: each video frame can produce both road-marking detections and a vehicle-facing decision object for autonomous/ADAS integration.

## Features

- Image, video, and webcam input
- HSV/Lab chromaticity segmentation for road paint colors
- Dedicated color-detection backend with sensitive, balanced, and strict profiles
- Per-color enable/disable controls and optional CLAHE contrast enhancement
- Road region-of-interest masking
- Morphological mask cleanup
- Contour extraction with bounding boxes, centroids, color, shape label, confidence, HSV mean, and normalized RGB chromaticity
- Overlay image, binary mask, JSON, and CSV output
- Streamlit web interface for quick tuning
- Dashboard UI with preview, detection, color coverage, decision, and export tabs
- Vehicle decision layer for video/webcam frames
- Steering hint, speed hint, risk level, action, reasons, and detected road-signal context
- Temporal road-pattern recognition for video input

## Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run The Web App

```bash
streamlit run app.py
```

Upload a road image or select one from `test/images`.

Large local datasets are not tracked in Git. Place CeyMo-style images under `test/images` or `train/images` when running the app locally.

If the local `venv` launcher is broken, use the included PowerShell launcher:

```powershell
.\run_app.ps1
```

## Process An Image

```bash
python -m lane_mark_chromaticity.cli image test\images\142.jpg --output-dir outputs
```

Useful detector options:

```bash
python -m lane_mark_chromaticity.cli image test\images\142.jpg --detector-profile sensitive --colors white yellow --use-clahe
```

Available detector profiles are `sensitive`, `balanced`, and `strict`.

With the included launcher:

```powershell
.\run_cli.ps1 image test\images\142.jpg --output-dir outputs
```

## Process A Video

```bash
python -m lane_mark_chromaticity.cli video path\to\road_video.mp4 --output-dir outputs
```

The video command writes `lanemark_overlay.mp4` with a decision HUD, `video_analysis.json` with per-frame detections/decisions/patterns, and `pattern_summary.json` with the dominant road pattern across the video.

## Run Webcam

```bash
python -m lane_mark_chromaticity.cli webcam --camera 0
```

Press `q` to stop the webcam preview.

## YOLOv8 Segmentation Workflow

Convert CeyMo polygon annotations to YOLOv8 segmentation labels:

```powershell
.\run_cli.ps1 export-yolo --source-root . --output-dir yolo_dataset --split test --copy-images
```

Train a YOLOv8 segmentation model after PyTorch native dependencies are working:

```powershell
.\run_cli.ps1 train-yolo --data yolo_dataset\ceymo-seg.yaml --model yolov8n-seg.pt --epochs 50 --image-size 960
```

Run YOLOv8 inference with trained weights:

```powershell
.\run_cli.ps1 yolo-predict --weights runs\segment\train\weights\best.pt --source test\images --output-dir outputs_yolo
```

## Output

For image input, the app writes:

- `<name>_overlay.jpg`
- `<name>_decision_overlay.jpg`
- `<name>_mask.png`
- `<name>_detections.json`
- `<name>_detections.csv`
- `<name>_vehicle_decision.json`

Each detection includes paint color, approximate marking type, confidence, area, bounding box, centroid, normalized RGB chromaticity, mean HSV, and contour points.

Video analysis also includes decision records:

- `action`: `keep_lane`, `correct_lane_position`, `prepare_to_stop`, `follow_marking`, `slow_and_verify`, or `proceed_with_caution`
- `steering_hint`: `hold_center`, `steer_left`, or `steer_right`
- `target_speed`: `normal` or `slow`
- `risk_level`: `low`, `medium`, or `high`
- `signals`: road context such as stop/crosswalk, arrows, yellow caution markings, red warning markings, or missing markings
- `reasons`: human-readable explanation for the decision

Video pattern records include:

- `pattern`: recognized pattern such as `crosswalk_or_stop_bar_pattern`, `direction_arrow_or_lane_symbol_pattern`, `yellow_caution_or_lane_boundary_pattern`, `centered_lane_corridor_pattern`, or `single_lane_boundary_pattern`
- `recommended_action`: vehicle-oriented action derived from the pattern
- `evidence`: what markings caused the pattern label
- `pattern_summary.json`: dominant pattern, confidence, frame count, pattern counts, color counts, and recommended video-level action

## Notes

This is a chromaticity-first classical computer vision baseline. It is fast and works without training, but faded markings, night scenes, strong shadows, and unusual pavement can still need threshold tuning or a learned detector such as YOLOv8/Mask R-CNN on top of this pipeline.

The current workspace already contains Python packages under `venv\Lib\site-packages`, but `venv\Scripts\python.exe` points to a removed Windows Store Python. The provided `run_app.ps1` and `run_cli.ps1` launch through the bundled Codex Python and reuse those installed packages.
