from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Detection:
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float
    label: str

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class TrackingProfile:
    x_p: float
    x_d: float
    y_p: float
    y_d: float
    forward_scale: float
    area_thresholds: Tuple[float, float, float, float, float, float]
    max_yaw_speed: int = 100
    max_vertical_speed: int = 60


FACE_PROFILE = TrackingProfile(
    x_p=0.3,
    x_d=0.2,
    y_p=0.2,
    y_d=0.2,
    forward_scale=1.0,
    area_thresholds=(
        20000 / (416 * 416),
        12000 / (416 * 416),
        7000 / (416 * 416),
        6000 / (416 * 416),
        4000 / (416 * 416),
        2000 / (416 * 416),
    ),
)

PERSON_PROFILE = TrackingProfile(
    x_p=0.3,
    x_d=0.2,
    y_p=0.3,
    y_d=0.2,
    forward_scale=2.0,
    area_thresholds=(
        80000 / (416 * 416),
        48000 / (416 * 416),
        28000 / (416 * 416),
        24000 / (416 * 416),
        16000 / (416 * 416),
        8000 / (416 * 416),
    ),
)


@dataclass(frozen=True)
class ControlCommand:
    left_right: int
    forward_backward: int
    up_down: int
    yaw: int
    x_error: Optional[float]
    y_error: Optional[float]
    area_ratio: float


def _clamp(value: float, lower: int, upper: int) -> int:
    return int(max(lower, min(upper, value)))


def compute_command(
    detection: Optional[Detection],
    frame_width: int,
    frame_height: int,
    profile: TrackingProfile,
    previous_error: Optional[Tuple[float, float]] = None,
) -> ControlCommand:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")

    if detection is None:
        return ControlCommand(0, 0, 0, 0, None, None, 0.0)

    x_error = detection.center_x - frame_width / 2
    y_error = frame_height / 2 - detection.center_y
    if previous_error is None:
        previous_error = (x_error, y_error)

    yaw = _clamp(
        profile.x_p * x_error + profile.x_d * (x_error - previous_error[0]),
        -profile.max_yaw_speed,
        profile.max_yaw_speed,
    )
    up_down = _clamp(
        profile.y_p * y_error + profile.y_d * (y_error - previous_error[1]),
        -profile.max_vertical_speed,
        profile.max_vertical_speed,
    )

    area_ratio = detection.area / (frame_width * frame_height)
    large, medium, neutral, near, far, very_small = profile.area_thresholds
    if area_ratio > large:
        forward_backward = int(-30 * profile.forward_scale)
    elif area_ratio > medium:
        forward_backward = int(-15 * profile.forward_scale)
    elif area_ratio > neutral or area_ratio == 0:
        forward_backward = 0
    elif area_ratio > near:
        forward_backward = int(15 * profile.forward_scale)
    elif area_ratio > far:
        forward_backward = int(20 * profile.forward_scale)
    elif area_ratio > very_small:
        forward_backward = 0
    elif area_ratio > 0:
        forward_backward = int(30 * profile.forward_scale)
    else:
        forward_backward = 0

    return ControlCommand(
        left_right=0,
        forward_backward=forward_backward,
        up_down=up_down,
        yaw=yaw,
        x_error=x_error,
        y_error=y_error,
        area_ratio=area_ratio,
    )
