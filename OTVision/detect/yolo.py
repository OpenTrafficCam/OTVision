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
from pathlib import Path
from time import perf_counter
from typing import Generator

import torch
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

from OTVision.abstraction.pipes_and_filter import Filter
from OTVision.application.detect.detected_frame_factory import DetectedFrameFactory
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.config import DetectConfig
from OTVision.domain.detection import DetectedFrame, Detection
from OTVision.domain.frame import Frame, FrameKeys
from OTVision.domain.object_detection import ObjectDetector, ObjectDetectorFactory
from OTVision.helpers.log import LOGGER_NAME

DISPLAYMATRIX = "DISPLAYMATRIX"

log = logging.getLogger(LOGGER_NAME)


class YoloDetectionConverter:
    """Converts raw YOLO model detections into standardized detection objects."""

    def convert(
        self,
        raw_detections: Boxes,
        normalized: bool,
        classification_mapping: dict[int, str],
    ) -> list[Detection]:
        """Converts raw detection data into a list of Detection objects.

        Args:
            raw_detections: The YOLO detection data.
            normalized: A boolean indicating whether the bounding box coordinates are
                normalized or not.
            classification_mapping: A dictionary mapping integer class indices to their
                corresponding class names.

        Returns:
            list[Detection]: A list of Detection objects containing information for
                each detection.
        """
        bboxes = raw_detections.xywhn if normalized else raw_detections.xywh
        detections: list[Detection] = []
        for bbox, class_idx, confidence in zip(
            bboxes, raw_detections.cls, raw_detections.conf
        ):
            _class_idx = int(class_idx.item())
            detections.append(
                self._create_detection(
                    bbox=bbox,
                    classification=classification_mapping[_class_idx],
                    confidence=confidence.item(),
                )
            )
        return detections

    def _create_detection(
        self,
        bbox: torch.Tensor,
        classification: str,
        confidence: float,
    ) -> Detection:
        x, y, width, height = bbox.tolist()

        return Detection(
            label=classification,
            conf=confidence,
            x=x - width / 2,
            y=y - height / 2,
            w=width,
            h=height,
        )


class YoloDetector(ObjectDetector, Filter[Frame, DetectedFrame]):
    """Wrapper to YOLO object detection model.

    Args:
        get_current_config (GetCurrentConfig): use case to get current configuration.
    """

    @property
    def config(self) -> DetectConfig:
        return self._get_current_config.get().detect

    @property
    def classifications(self) -> dict[int, str]:
        """The model's classes that it is able to predict.

        Returns:
            dict[int, str]: the classes
        """
        return (
            self._model.names
            if self._model.names is not None
            else self._model.predictor.model.names
        )

    def __init__(
        self,
        model: YOLO,
        get_current_config: GetCurrentConfig,
        detection_converter: YoloDetectionConverter,
        detected_frame_factory: DetectedFrameFactory,
    ) -> None:
        self._model = model
        self._get_current_config = get_current_config
        self._detection_converter = detection_converter
        self._detected_frame_factory = detected_frame_factory

    def filter(
        self, pipe: Generator[Frame, None, None]
    ) -> Generator[DetectedFrame, None, None]:
        return self.detect(pipe)

    def detect(
        self, frames: Generator[Frame, None, None]
    ) -> Generator[DetectedFrame, None, None]:
        for frame in tqdm(frames, desc="Detected frames", unit=" frames"):
            yield self._predict(frame)

    def _predict(self, frame: Frame) -> DetectedFrame:
        if frame[FrameKeys.data] is None:
            return self._create_empty_detection(frame)

        return self._process_frame(frame)

    def _process_frame(self, frame: Frame) -> DetectedFrame:
        """Process a single frame and return detected objects."""
        model_predictions = self._model.predict(
            source=frame[FrameKeys.data],
            conf=self.config.confidence,
            iou=self.config.iou,
            half=self.config.half_precision,
            imgsz=self.config.img_size,
            device=0 if torch.cuda.is_available() else "cpu",
            stream=False,
            verbose=False,
            agnostic_nms=True,
        )

        for prediction in model_predictions:
            return self._create_detection_from_boxes(frame, prediction.boxes)

        # Return empty detection if no predictions
        return self._create_empty_detection(frame)

    def _create_detection_from_boxes(self, frame: Frame, boxes: Boxes) -> DetectedFrame:
        """Convert raw detection boxes to a DetectedFrame with detected objects."""
        detections = self._detection_converter.convert(
            boxes,
            normalized=self.config.normalized,
            classification_mapping=self.classifications,
        )
        return self._detected_frame_factory.create(frame, detections=detections)

    def _create_empty_detection(self, frame: Frame) -> DetectedFrame:
        """Create a DetectedFrame with no detections."""
        return self._detected_frame_factory.create(frame, detections=[])


class YoloFactory(ObjectDetectorFactory):
    """
    A factory class responsible for creating YOLO object detection instances.

    This class provides functionalities for initializing a YOLO model using specified
    configurations and dependencies. It supports custom or pretrained model weights
    and handles the setup of various model parameters.

    Args:
        get_current_config (GetCurrentConfig): Use case to get current configuration.
        detection_converter (YoloDetectionConverter): Convert yolo detection results
            to `Detection` objects.
        detected_frame_factory (DetectedFrameFactory): Factory to create`DetectedFrame`
            objects.
    """

    def __init__(
        self,
        get_current_config: GetCurrentConfig,
        detection_converter: YoloDetectionConverter,
        detected_frame_factory: DetectedFrameFactory,
    ) -> None:
        self._get_current_config = get_current_config
        self._detection_converter = detection_converter
        self._detected_frame_factory = detected_frame_factory

    def create(self, config: DetectConfig) -> ObjectDetector:
        """
        Creates an object detection model using YOLO with the specified configuration.

        This method initializes a YOLOv8 model using the parameters provided in the
        DetectConfig instance. It determines whether the provided weights reference
        a custom file or a pretrained model. The method configures the model with
        confidence threshold, intersection over union (IoU) threshold, image size,
        and other properties as defined in the input configuration. Additionally,
        it logs the loading time and specifies whether the model is utilizing CUDA
        or operating in CPU mode. After successful initialization, the method
        returns the prepared object detection model.

        Args:
            config (DetectConfig): A configuration instance containing YOLO
                model parameters including weights path, thresholds, image size,
                and other initialization settings.

        Returns:
            ObjectDetector: An initialized YOLO object detection model ready
                for inference.
        """
        weights = config.yolo_config.weights
        log.info(f"Try loading model {weights}")
        t1 = perf_counter()
        is_custom = Path(weights).is_file()
        model = YoloDetector(
            model=self._load_model(weights),
            get_current_config=self._get_current_config,
            detection_converter=self._detection_converter,
            detected_frame_factory=self._detected_frame_factory,
        )
        t2 = perf_counter()

        model_source = "Custom" if is_custom else "Pretrained"
        model_type = "CUDA" if torch.cuda.is_available() else "CPU"
        runtime = round(t2 - t1)
        log.info(f"{model_source} {model_type} model loaded in {runtime} sec")

        model_success_msg = f"Model {weights} prepared"
        log.info(model_success_msg)

        return model

    def _load_model(self, weights: str | Path) -> YOLO:
        """Load a custom trained or a pretrained YOLOv8 model.

        Args:
            weights (str | Path): Either path to custom model weights or pretrained
                model.

        Returns:
            YOLOv8: the YOLOv8 model.

        """
        model = YOLO(model=weights, task="detect")
        return model
