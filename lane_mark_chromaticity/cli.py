from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .pipeline import DetectionConfig, analyze_image, combined_mask, load_image, process_image_file
from .patterns import RoadPatternTracker
from .vehicle_decision import draw_decision_hud, make_vehicle_decision
from .yolo_tools import convert_ceymo_to_yolo, predict_yolo, train_yolo


def _config_from_args(args: argparse.Namespace) -> DetectionConfig:
    color_args = getattr(args, "colors", ["white", "yellow", "red", "blue"])
    if isinstance(color_args, str):
        color_args = [color_args]
    enabled_colors = tuple(
        color.strip().lower()
        for value in color_args
        for color in value.split(",")
        if color.strip()
    )
    return DetectionConfig(
        resize_width=args.resize_width,
        roi_top_ratio=args.roi_top_ratio,
        min_area=args.min_area,
        morph_kernel=args.morph_kernel,
        mask_blur=args.mask_blur,
        detector_profile=args.detector_profile,
        enabled_colors=enabled_colors,
        use_clahe=args.use_clahe,
    )


def run_image(args: argparse.Namespace) -> None:
    config = _config_from_args(args)
    result = process_image_file(args.input, args.output_dir, config)
    output_dir = Path(args.output_dir)
    image = load_image(args.input)
    overlay, _, detections = analyze_image(image, config)
    decision = make_vehicle_decision(detections, overlay.shape)
    decision_overlay = draw_decision_hud(overlay, decision)
    decision_overlay_path = output_dir / f"{Path(args.input).stem}_decision_overlay.jpg"
    decision_json_path = output_dir / f"{Path(args.input).stem}_vehicle_decision.json"
    cv2.imwrite(str(decision_overlay_path), decision_overlay)
    decision_json_path.write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")
    print(f"Detections: {result['detections']}")
    print(f"Overlay: {result['overlay']}")
    print(f"Mask: {result['mask']}")
    print(f"JSON: {result['json']}")
    print(f"CSV: {result['csv']}")
    print(f"Decision overlay: {decision_overlay_path}")
    print(f"Decision JSON: {decision_json_path}")


def run_video(args: argparse.Namespace) -> None:
    config = _config_from_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    writer = None
    frame_index = 0
    all_records = []
    tracker = RoadPatternTracker(window_size=args.pattern_window)
    output_video = output_dir / "lanemark_overlay.mp4"

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        overlay, masks, detections = analyze_image(frame, config)
        decision = make_vehicle_decision(detections, overlay.shape)
        frame_pattern = tracker.update(frame_index, detections, decision, overlay.shape)
        overlay = draw_decision_hud(overlay, decision)
        cv2.putText(
            overlay,
            f"PATTERN: {frame_pattern.pattern}",
            (16, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if writer is None:
            height, width = overlay.shape[:2]
            writer = cv2.VideoWriter(
                str(output_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
        writer.write(overlay)
        for detection in detections:
            record = detection.__dict__.copy()
            record["frame"] = frame_index
            all_records.append(record)
        all_records.append(
            {
                "frame": frame_index,
                "type": "vehicle_decision",
                **decision.to_dict(),
            }
        )
        all_records.append(
            {
                "frame": frame_index,
                "type": "road_pattern",
                **frame_pattern.to_dict(),
            }
        )
        frame_index += 1

    capture.release()
    if writer is not None:
        writer.release()

    summary = tracker.summary()
    (output_dir / "video_analysis.json").write_text(
        json.dumps(all_records, indent=2),
        encoding="utf-8",
    )
    (output_dir / "pattern_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2),
        encoding="utf-8",
    )
    print(f"Processed frames: {frame_index}")
    print(f"Overlay video: {output_video}")
    print(f"JSON: {output_dir / 'video_analysis.json'}")
    print(f"Pattern summary: {output_dir / 'pattern_summary.json'}")


def run_webcam(args: argparse.Namespace) -> None:
    config = _config_from_args(args)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    print("Webcam started. Press q to exit.")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        overlay, masks, detections = analyze_image(frame, config)
        decision = make_vehicle_decision(detections, overlay.shape)
        overlay = draw_decision_hud(overlay, decision)
        mask = cv2.cvtColor(combined_mask(masks), cv2.COLOR_GRAY2BGR)
        preview = cv2.hconcat([overlay, mask])
        cv2.putText(
            preview,
            f"detections: {len(detections)}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("LaneMark Chromaticity AI", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def run_export_yolo(args: argparse.Namespace) -> None:
    yaml_path = convert_ceymo_to_yolo(
        source_root=args.source_root,
        output_root=args.output_dir,
        split=args.split,
        copy_images=args.copy_images,
    )
    print(f"YOLO dataset YAML: {yaml_path}")


def run_train_yolo(args: argparse.Namespace) -> None:
    result = train_yolo(args.data, args.model, args.epochs, args.image_size)
    print(result)


def run_yolo_predict(args: argparse.Namespace) -> None:
    result = predict_yolo(args.weights, args.source, args.output_dir)
    print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LaneMark Chromaticity AI")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--resize-width", type=int, default=1280)
        subparser.add_argument("--roi-top-ratio", type=float, default=0.58)
        subparser.add_argument("--min-area", type=int, default=400)
        subparser.add_argument("--morph-kernel", type=int, default=5)
        subparser.add_argument("--mask-blur", type=int, default=3)
        subparser.add_argument(
            "--detector-profile",
            default="balanced",
            choices=["sensitive", "balanced", "strict"],
        )
        subparser.add_argument("--colors", nargs="+", default=["white", "yellow", "red", "blue"])
        subparser.add_argument("--use-clahe", action="store_true")

    image = subparsers.add_parser("image", help="Process one road image")
    image.add_argument("input")
    image.add_argument("--output-dir", default="outputs")
    add_common(image)
    image.set_defaults(func=run_image)

    video = subparsers.add_parser("video", help="Process a road video")
    video.add_argument("input")
    video.add_argument("--output-dir", default="outputs")
    video.add_argument("--pattern-window", type=int, default=12)
    add_common(video)
    video.set_defaults(func=run_video)

    webcam = subparsers.add_parser("webcam", help="Run live camera preview")
    webcam.add_argument("--camera", type=int, default=0)
    add_common(webcam)
    webcam.set_defaults(func=run_webcam)

    export_yolo = subparsers.add_parser(
        "export-yolo",
        help="Convert CeyMo polygon annotations to YOLOv8 segmentation format",
    )
    export_yolo.add_argument("--source-root", default=".")
    export_yolo.add_argument("--output-dir", default="yolo_dataset")
    export_yolo.add_argument("--split", default="test", choices=["train", "test"])
    export_yolo.add_argument("--copy-images", action="store_true")
    export_yolo.set_defaults(func=run_export_yolo)

    train = subparsers.add_parser("train-yolo", help="Train a YOLOv8 segmentation model")
    train.add_argument("--data", required=True)
    train.add_argument("--model", default="yolov8n-seg.pt")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--image-size", type=int, default=960)
    train.set_defaults(func=run_train_yolo)

    predict = subparsers.add_parser("yolo-predict", help="Run YOLOv8 inference")
    predict.add_argument("--weights", required=True)
    predict.add_argument("--source", required=True)
    predict.add_argument("--output-dir", default="outputs_yolo")
    predict.set_defaults(func=run_yolo_predict)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
