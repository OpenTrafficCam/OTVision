"""
OTVision module to detect objects using yolov5
"""

# Copyright (C) 2022 OpenTrafficCam Contributors
# <https://github.com/OpenTrafficCam
# <team@opentrafficcam.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from time import perf_counter
from typing import Callable, Generator

import av
import torch
from tqdm import tqdm
from ultralytics import YOLO as YOLOv8
from ultralytics.engine.results import Boxes

from OTVision.detect.plugin_av.rotate_frame import AvVideoFrameRotator
from OTVision.helpers import video
from OTVision.helpers.log import LOGGER_NAME
from OTVision.track.preprocess import Detection

DISPLAYMATRIX = "DISPLAYMATRIX"

log = logging.getLogger(LOGGER_NAME)


class VideoFiletypeNotSupportedError(Exception):
    pass


class VideoFoundError(Exception):
    pass


class YOLOv5ModelNotFoundError(Exception):
    pass


class ObjectDetection(ABC):
    @abstractmethod
    def detect(
        self,
        video: Path,
        detect_start: int | None = None,
        detect_end: int | None = None,
    ) -> list[list[Detection]]:
        """Runs object detection on a video.
        Args:
            video (Path): the path to the video.
            detect_start (int | None, optional): Start of the detection range
                expressed in frames.
            detect_end (int | None, optional): End of the detection range
                expressed in frames. Defaults to None.

        Returns:
            list[list[Detection]]: nested list of detections. First level is frames,
            second level is detections within frame
        """
        pass


class Yolov8(ObjectDetection):
    """Wrapper to YOLOv8 object detection model.

    Args:
        weights (str | Path): Custom model weights for prediction.
        model: (YOLOv8):  the YOLOv8 model to use for prediction.
        confidence (float): the confidence threshold.
        iou (float): the IOU threshold.
        img_size (int): the YOLOv8 img size.
        half_precision (bool): Whether to use half precision (FP16) for inference speed
            up.
        normalized (bool): Whether the bounding boxes are to be returned normalized.
        frame_rotator: (AvVideoFrameRotator): use case to use rotate video frames.
        get_number_of_frames: (Callable[[Path], int]): function to get the total number
            of frames of a video.
    """

    def __init__(
        self,
        weights: str | Path,
        model: YOLOv8,
        confidence: float,
        iou: float,
        img_size: int,
        half_precision: bool,
        normalized: bool,
        frame_rotator: AvVideoFrameRotator,
        get_number_of_frames: Callable[[Path], int] = video.get_number_of_frames,
    ) -> None:
        self.weights = weights
        self.model = model
        self.confidence = confidence
        self.iou = iou
        self.img_size = img_size
        self.half_precision = half_precision
        self.normalized = normalized
        self._frame_rotator = frame_rotator
        self._get_number_of_frames = get_number_of_frames

    @property
    def classifications(self) -> dict[int, str]:
        """The model's classes that it is able to predict.

        Returns:
            dict[int, str]: the classes
        """
        return (
            self.model.names
            if self.model.names is not None
            else self.model.predictor.model.names
        )

    def detect(
        self, file: Path, detect_start: int | None = None, detect_end: int | None = None
    ) -> list[list[Detection]]:
        """Run object detection on video and return detection result.

        Args:
            file (Path): the video to run object detection on.
            detect_start (int | None, optional): Start of the detection range in frames.
                Defaults to None.
            detect_end (int | None, optional): End of the detection range in frames.
                Defaults to None.

        Returns:
            list[list[Detection]]: the detections for each frame in the video
        """
        frames: list[list[Detection]] = []
        length = self._get_number_of_frames(file)
        for prediction_result in tqdm(
            self._predict(file, detect_start, detect_end),
            desc="Detected frames",
            unit=" frames",
            total=length,
        ):
            frames.append(prediction_result)

        return frames

    def _predict(
        self, video: Path, detect_start: int | None, detect_end: int | None
    ) -> Generator[list[Detection], None, None]:
        start = 0
        if detect_start is not None:
            start = detect_start

        with av.open(str(video.absolute())) as container:
            container.streams.video[0].thread_type = "AUTO"
            side_data = container.streams.video[0].side_data
            for frame_number, frame in enumerate(container.decode(video=0), start=1):
                if start <= frame_number and (
                    detect_end is None or frame_number < detect_end
                ):
                    rotated_image = self._frame_rotator.rotate(frame, side_data)
                    results = self.model.predict(
                        source=rotated_image,
                        conf=self.confidence,
                        iou=self.iou,
                        half=self.half_precision,
                        imgsz=self.img_size,
                        device=0 if torch.cuda.is_available() else "cpu",
                        stream=False,
                        verbose=False,
                        agnostic_nms=True,
                    )
                    for result in results:
                        yield self._parse_detections(result.boxes)
                else:
                    yield []

    def _parse_detections(self, detection_result: Boxes) -> list[Detection]:
        bboxes = detection_result.xywhn if self.normalized else detection_result.xywh
        detections: list[Detection] = []
        for bbox, class_idx, confidence in zip(
            bboxes, detection_result.cls, detection_result.conf
        ):
            detections.append(
                self._parse_detection(bbox, int(class_idx.item()), confidence.item())
            )
        return detections

    def _parse_detection(
        self, bbox: torch.Tensor, class_idx: int, confidence: float
    ) -> Detection:
        x, y, width, height = bbox.tolist()
        classification = self.classifications[class_idx]

        return Detection(
            label=classification,
            conf=confidence,
            x=x - width / 2,
            y=y - height / 2,
            w=width,
            h=height,
        )


def create_model(
    weights: str | Path,
    confidence: float,
    iou: float,
    img_size: int,
    half_precision: bool,
    normalized: bool,
) -> Yolov8:
    """Loads a custom trained or a pretrained YOLOv8 mode.

    Args:
        weights (str | Path): Either path to custom model weights or pretrained model
            name, i.e. 'yolov8s', 'yolov8m'.
        confidence (float): the confidence threshold.
        iou (float): the IOU threshold
        img_size (int): the YOLOv8 image size
        half_precision (bool): Whether to use half precision (FP16) for inference speed
            up.
        normalized (bool): Whether the bounding boxes are to be returned normalized

    Returns:
        Yolov8: the YOLOv8 model
    """
    log.info(f"Try loading model {weights}")
    t1 = perf_counter()
    is_custom = Path(weights).is_file()
    model = Yolov8(
        weights=weights,
        model=_load_model(weights),
        confidence=confidence,
        iou=iou,
        img_size=img_size,
        half_precision=half_precision,
        normalized=normalized,
        frame_rotator=AvVideoFrameRotator(),
    )
    t2 = perf_counter()

    model_source = "Custom" if is_custom else "Pretrained"
    model_type = "CUDA" if torch.cuda.is_available() else "CPU"
    runtime = round(t2 - t1)
    log.info(f"{model_source} {model_type} model loaded in {runtime} sec")

    model_success_msg = f"Model {weights} prepared"
    log.info(model_success_msg)

    return model


def _load_model(weights: str | Path) -> YOLOv8:
    """Load a custom trained or a pretrained YOLOv8 model.

    Args:
        weights (str | Path): Either path to custom model weights or pretrained model
            name, i.e. 'yolov8s', 'yolov8m'.

    Returns:
        YOLOv8: the YOLOv8 model.

    """
    model = YOLOv8(model=weights, task="detect")
    return model
