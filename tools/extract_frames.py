import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from a video for manual annotation."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Save every Nth frame (default: every frame).",
    )
    parser.add_argument("--size", type=int, default=416)
    return parser.parse_args()


def extract_frames(video: Path, output_dir: Path, every: int, size: int) -> int:
    if every < 1:
        raise ValueError("--every must be at least 1.")
    if size < 1:
        raise ValueError("--size must be positive.")
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % every == 0:
                resized = cv2.resize(
                    frame, (size, size), interpolation=cv2.INTER_AREA
                )
                destination = output_dir / f"frame_{frame_index:06d}.jpg"
                if not cv2.imwrite(str(destination), resized):
                    raise OSError(f"Could not write frame: {destination}")
                saved += 1
            frame_index += 1
    finally:
        capture.release()

    print(f"Saved {saved} frames to {output_dir.resolve()}")
    return saved


if __name__ == "__main__":
    args = parse_args()
    extract_frames(args.video, args.output_dir, args.every, args.size)
