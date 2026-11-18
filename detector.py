from pathlib import Path
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch

from tracking_control import Detection


REPOSITORY_ROOT = Path(__file__).resolve().parent
YOLOV5_ROOT = REPOSITORY_ROOT / "yolov5"
if not YOLOV5_ROOT.is_dir():
    raise FileNotFoundError(
        "YOLOv5 submodule is missing. Run: "
        "git submodule update --init --recursive"
    )
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync


def resolve_model(
    target: str, weights: Optional[Path], data: Optional[Path]
) -> Tuple[Path, Optional[Path]]:
    if target not in ("face", "person"):
        raise ValueError(f"Unsupported target: {target}")
    if target == "face" and weights is None:
        raise ValueError(
            "Face tracking needs custom face-model weights; pass --weights PATH. "
            "The weights are not included in this repository."
        )
    resolved_weights = weights or REPOSITORY_ROOT / "weights" / "yolov5s.pt"
    resolved_data = data
    if resolved_data is None and target == "person":
        resolved_data = YOLOV5_ROOT / "data" / "coco128.yaml"
    return resolved_weights, resolved_data


class YoloDetector:
    def __init__(
        self,
        weights: Path,
        data: Optional[Path] = None,
        device: str = "",
        image_size: int = 416,
        confidence: float = 0.4,
        iou_threshold: float = 0.45,
    ) -> None:
        self.device = select_device(device)
        self.weights = Path(weights).expanduser().resolve()
        self.data = Path(data).expanduser().resolve() if data else None
        if self.data is not None and not self.data.is_file():
            raise FileNotFoundError(f"Dataset configuration not found: {self.data}")
        if not 0 < confidence <= 1:
            raise ValueError("Confidence threshold must be in (0, 1].")
        if not 0 < iou_threshold <= 1:
            raise ValueError("IoU threshold must be in (0, 1].")

        self.model = DetectMultiBackend(
            self.weights,
            device=self.device,
            dnn=False,
            data=str(self.data) if self.data else None,
            fp16=False,
        )
        self.stride = int(self.model.stride)
        self.image_size = check_img_size(image_size, s=self.stride)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.names = self.model.names
        self.model.warmup(imgsz=(1, 3, self.image_size, self.image_size))

    def _class_name(self, class_id: int) -> str:
        return str(self.names[class_id])

    @staticmethod
    def _matches_target(label: str, target: str) -> bool:
        normalized = label.casefold()
        if target == "face":
            return "face" in normalized
        return normalized == "person"

    @torch.no_grad()
    def detect(
        self, frame: np.ndarray, target: str
    ) -> Tuple[List[Detection], np.ndarray, float]:
        if target not in ("face", "person"):
            raise ValueError(f"Unsupported target: {target}")
        if frame is None or frame.size == 0:
            raise ValueError("Cannot run detection on an empty frame.")

        image = letterbox(
            frame, self.image_size, stride=self.stride, auto=True
        )[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image)
        tensor = torch.from_numpy(image).to(self.device).float() / 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        start = time_sync()
        prediction = self.model(tensor, augment=False, visualize=False)
        prediction = non_max_suppression(
            prediction,
            self.confidence,
            self.iou_threshold,
            classes=None,
            agnostic=False,
            max_det=100,
        )
        inference_ms = (time_sync() - start) * 1000

        annotated = frame.copy()
        annotator = Annotator(annotated, line_width=2, example="person, face")
        detections: List[Detection] = []
        for det in prediction:
            if not len(det):
                continue
            det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, confidence, class_id in det:
                label = self._class_name(int(class_id))
                if not self._matches_target(label, target):
                    continue
                x1, y1, x2, y2 = (float(value) for value in xyxy)
                detections.append(
                    Detection(
                        center_x=(x1 + x2) / 2,
                        center_y=(y1 + y2) / 2,
                        width=max(0.0, x2 - x1),
                        height=max(0.0, y2 - y1),
                        confidence=float(confidence),
                        label=label,
                    )
                )
                annotator.box_label(
                    xyxy,
                    f"{label} {float(confidence):.2f}",
                    color=colors(int(class_id), True),
                )

        return detections, annotator.result(), inference_ms
