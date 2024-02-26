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
from typing import Generator

import torch
from tqdm import tqdm
from ultralytics import YOLO as YOLOv8
from ultralytics.engine.results import Boxes, Results

from OTVision.config import (
    CONF,
    CONFIG,
    DETECT,
    HALF_PRECISION,
    IMG_SIZE,
    IOU,
    NORMALIZED,
    WEIGHTS,
    YOLO,
)
from OTVision.helpers import video
from OTVision.helpers.log import LOGGER_NAME
from OTVision.track.preprocess import Detection

log = logging.getLogger(LOGGER_NAME)


class VideoFiletypeNotSupportedError(Exception):
    pass


class VideoFoundError(Exception):
    pass


class YOLOv5ModelNotFoundError(Exception):
    pass


class ObjectDetection(ABC):
    @abstractmethod
    def detect(self, video: Path) -> list[list[Detection]]:
        """Runs object detection on a video.
        Args:
            video (Path): the path to the video.

        Returns:
            list[list[Detection]]: nested list of detections. First level is frames,
            second level is detections within frame
        """
        pass


class Yolov8(ObjectDetection):
    """Wrapper to YOLOv8 object detection model.

    Args:
        weights (str | Path): Either path to custom model weights or pretrained model
            name, i.e. 'yolov8s', 'yolov8m'.
        confidence (float): the confidence threshold
        iou (float): the IOU threshold
        img_size (int): the YOLOv8 img size
        half_precision (bool): Whether to use half precision (FP16) for inference speed
            up.
        normalized (bool): Whether the bounding boxes are to be returned normalized
    """

    def __init__(
        self,
        weights: str | Path,
        confidence: float,
        iou: float,
        img_size: int,
        half_precision: bool,
        normalized: bool,
    ) -> None:
        self.weights = weights
        self.confidence = confidence
        self.iou = iou
        self.img_size = img_size
        self.half_precision = half_precision
        self.normalized = normalized

        self.model: YOLOv8 = self._load_model()

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

    def detect(self, file: Path) -> list[list[Detection]]:
        """Run object detection on video and return detection result.

        Args:
            video (Path): the video to run object detection on

        Returns:
            list[list[Detection]]: the detections for each frame in the video
        """
        frames: list[list[Detection]] = []
        length = video.get_number_of_frames(file)
        for prediction_result in tqdm(
            self._predict(file),
            desc="Detected frames",
            unit="frames",
            total=length,
        ):
            frames.append(self._parse_detections(prediction_result.boxes))

        return frames

    def _load_model(self) -> YOLOv8:
        model = YOLOv8(model=self.weights, task="detect")
        return model

    def _predict(self, video: Path) -> Generator[Results, None, None]:
        return self.model.predict(
            source=video,
            conf=self.confidence,
            iou=self.iou,
            half=self.half_precision,
            imgsz=self.img_size,
            device=0 if torch.cuda.is_available() else "cpu",
            stream=True,
            verbose=False,
            batch=-1,
            agnostic_nms=True,
        )

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


def loadmodel(
    weights: str | Path = CONFIG[DETECT][YOLO][WEIGHTS],
    confidence: float = CONFIG[DETECT][YOLO][CONF],
    iou: float = CONFIG[DETECT][YOLO][IOU],
    img_size: int = CONFIG[DETECT][YOLO][IMG_SIZE],
    half_precision: bool = CONFIG[DETECT][HALF_PRECISION],
    normalized: bool = CONFIG[DETECT][YOLO][NORMALIZED],
) -> Yolov8:
    """Loads a custom trained or a pretrained YOLOv8 mode.

    Args:
        weights (str | Path): Either path to custom model weights or pretrained model
            name, i.e. 'yolov8s', 'yolov8m'.
        confidence (float): the confidence threshold.
        iou (float): the IOU threshold
        image_size (int): the YOLOv8 image size
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
        confidence=confidence,
        iou=iou,
        img_size=img_size,
        half_precision=half_precision,
        normalized=normalized,
    )
    t2 = perf_counter()

    model_source = "Custom" if is_custom else "Pretrained"
    model_type = "CUDA" if torch.cuda.is_available() else "CPU"
    runtime = round(t2 - t1)
    log.info(f"{model_source} {model_type} model loaded in {runtime} sec")

    model_succes_msg = f"Model {model.weights} prepared"
    log.info(model_succes_msg)
    print(model_succes_msg)

    return model
