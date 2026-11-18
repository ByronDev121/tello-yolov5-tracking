import argparse
from pathlib import Path

import cv2

from detector import YoloDetector, resolve_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tracker on a webcam.")
    parser.add_argument("--target", choices=("face", "person"), default="person")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", default="")
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights, data = resolve_model(args.target, args.weights, args.data)
    detector = YoloDetector(
        weights=weights,
        data=data,
        device=args.device,
        image_size=args.image_size,
        confidence=args.confidence,
        iou_threshold=args.iou,
    )
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera stopped returning frames.")
            _, annotated, inference_ms = detector.detect(frame, args.target)
            cv2.putText(
                annotated,
                f"Inference: {inference_ms:.1f} ms",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow(f"Webcam {args.target} tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
