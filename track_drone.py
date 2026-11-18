import argparse
import csv
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
from djitellopy import Tello

from detector import YoloDetector, resolve_model
from tracking_control import (
    FACE_PROFILE,
    PERSON_PROFILE,
    ControlCommand,
    Detection,
    compute_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track a face or person and control a DJI Tello."
    )
    parser.add_argument("--target", choices=("face", "person"), default="person")
    parser.add_argument("--weights", type=Path, help="Model weights (.pt).")
    parser.add_argument("--data", type=Path, help="Optional YOLOv5 data YAML.")
    parser.add_argument("--device", default="", help="CUDA device, or empty for auto.")
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--fly", action="store_true", help="Take off and send flight commands.")
    parser.add_argument("--no-display", action="store_true", help="Disable the OpenCV window.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "tracking",
        help="Directory for CSV and plots.",
    )
    return parser.parse_args()


def choose_detection(detections: List[Detection]) -> Optional[Detection]:
    if not detections:
        return None
    return max(detections, key=lambda detection: detection.confidence)


def record_sample(
    samples: List[Dict[str, object]],
    command: ControlCommand,
    detection: Optional[Detection],
    inference_ms: float,
) -> None:
    samples.append(
        {
            "time_seconds": time.monotonic(),
            "is_detecting": int(detection is not None),
            "inference_ms": inference_ms,
            "x_error": command.x_error,
            "y_error": command.y_error,
            "forward_backward": command.forward_backward,
            "y_speed": command.up_down,
            "x_speed": command.yaw,
            "area_ratio": command.area_ratio,
        }
    )


def save_logs(samples: List[Dict[str, object]], output_dir: Path) -> None:
    if not samples:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "flight_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)

    plots = (
        ("is_detecting", "Is Inferring"),
        ("inference_ms", "Object Detection Time"),
        ("forward_backward", "Forward Backward"),
        ("x_speed", "X Speed (Yaw)"),
        ("y_speed", "Y Speed (Vertical)"),
        ("x_error", "X Error"),
        ("y_error", "Y Error"),
    )
    steps = range(len(samples))
    for column, title in plots:
        values = [sample[column] for sample in samples]
        figure, axis = plt.subplots()
        axis.plot(steps, values)
        axis.set(xlabel="Step", ylabel=title, title=title)
        figure.tight_layout()
        figure.savefig(output_dir / f"{column}.png")
        plt.close(figure)
    print(f"Saved flight log and plots to {output_dir.resolve()}")


def run(args: argparse.Namespace) -> None:
    if args.duration <= 0:
        raise ValueError("Duration must be positive.")
    weights, data = resolve_model(args.target, args.weights, args.data)
    detector = YoloDetector(
        weights=weights,
        data=data,
        device=args.device,
        image_size=args.image_size,
        confidence=args.confidence,
        iou_threshold=args.iou,
    )
    profile = FACE_PROFILE if args.target == "face" else PERSON_PROFILE
    tello = Tello()
    connected = False
    streaming = False
    airborne = False
    samples: List[Dict[str, object]] = []
    previous_error = None

    try:
        tello.connect()
        connected = True
        tello.streamon()
        streaming = True
        frame_read = tello.get_frame_read()

        if args.fly:
            airborne = True
            tello.takeoff()
            print("Tello is airborne. Press q in the video window to stop early.")
        else:
            print("Preview only: no takeoff or flight commands will be sent.")

        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            frame = frame_read.frame
            if frame is None:
                time.sleep(0.01)
                continue

            detections, annotated, inference_ms = detector.detect(frame, args.target)
            detection = choose_detection(detections)
            height, width = frame.shape[:2]
            command = compute_command(
                detection, width, height, profile, previous_error
            )
            previous_error = (
                (command.x_error, command.y_error)
                if detection is not None
                else None
            )
            if args.fly:
                tello.send_rc_control(
                    command.left_right,
                    command.forward_backward,
                    command.up_down,
                    command.yaw,
                )
            record_sample(samples, command, detection, inference_ms)

            if not args.no_display:
                cv2.imshow(f"Tello {args.target} tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        try:
            if connected and airborne:
                tello.send_rc_control(0, 0, 0, 0)
        finally:
            try:
                if airborne:
                    tello.land()
            finally:
                try:
                    if streaming:
                        tello.streamoff()
                finally:
                    cv2.destroyAllWindows()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_logs(samples, args.output_dir / timestamp)


if __name__ == "__main__":
    run(parse_args())
