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
from typing import Callable, Generator, Self

import av
import torch
from tqdm import tqdm
from ultralytics import YOLO as YOLOv8
from ultralytics.engine.results import Boxes

from OTVision.config import DetectConfig
from OTVision.detect.plugin_av.rotate_frame import AvVideoFrameRotator
from OTVision.domain.detect import ObjectDetector, ObjectDetectorFactory
from OTVision.helpers import video
from OTVision.helpers.log import LOGGER_NAME
from OTVision.helpers.video import convert_seconds_to_frames, get_fps
from OTVision.track.preprocess import Detection

DISPLAYMATRIX = "DISPLAYMATRIX"

log = logging.getLogger(LOGGER_NAME)


class VideoFiletypeNotSupportedError(Exception):
    pass


class VideoFoundError(Exception):
    pass


class YOLOv5ModelNotFoundError(Exception):
    pass


class YoloDetector(ObjectDetector):
    """Wrapper to YOLO object detection model.

    Args:
        model: (YOLOv8):  the YOLOv8 model to use for prediction.
        config (DetectConfig): the configuration to use for prediction.
        frame_rotator: (AvVideoFrameRotator): use case to use rotate video frames.
        get_number_of_frames: (Callable[[Path], int]): function to get the total number
            of frames of a video.
    """

    @property
    def config(self) -> DetectConfig:
        return self._config

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
        model: YOLOv8,
        frame_rotator: AvVideoFrameRotator,
        config: DetectConfig,
        get_number_of_frames: Callable[[Path], int] = video.get_number_of_frames,
    ) -> None:
        self._model = model
        self._frame_rotator = frame_rotator
        self._get_number_of_frames = get_number_of_frames
        self._config = config
        self.configure_with(config)

    def configure_with(self, config: DetectConfig) -> Self:
        self._config = config
        return self

    def _convert_seconds_to_frame(
        self, seconds: int | None, video_file: Path
    ) -> int | None:
        video_fps = get_fps(video_file)
        return convert_seconds_to_frames(seconds, video_fps)

    def detect(self, source: str) -> Generator[list[Detection], None, None]:
        video_source = Path(source)
        length = self._get_number_of_frames(video_source)
        for prediction_result in tqdm(
            self._predict(video_source),
            desc="Detected frames",
            unit=" frames",
            total=length,
        ):
            yield prediction_result

    def _predict(self, video_source: Path) -> Generator[list[Detection], None, None]:
        start = 0
        detect_start = self._convert_seconds_to_frame(
            self.config.detect_start, video_source
        )
        detect_end = self._convert_seconds_to_frame(
            self.config.detect_end, video_source
        )

        if detect_start is not None:
            start = detect_start

        with av.open(str(video_source.absolute())) as container:
            container.streams.video[0].thread_type = "AUTO"
            side_data = container.streams.video[0].side_data
            for frame_number, frame in enumerate(container.decode(video=0), start=1):
                if start <= frame_number and (
                    detect_end is None or frame_number < detect_end
                ):
                    rotated_image = self._frame_rotator.rotate(frame, side_data)
                    results = self._model.predict(
                        source=rotated_image,
                        conf=self.config.confidence,
                        iou=self.config.iou,
                        half=self.config.half_precision,
                        imgsz=self.config.img_size,
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
        bboxes = (
            detection_result.xywhn if self._config.normalized else detection_result.xywh
        )
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


class YoloFactory(ObjectDetectorFactory):

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
            config=config,
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

    def _load_model(self, weights: str | Path) -> YOLOv8:
        """Load a custom trained or a pretrained YOLOv8 model.

        Args:
            weights (str | Path): Either path to custom model weights or pretrained
                model.

        Returns:
            YOLOv8: the YOLOv8 model.

        """
        model = YOLOv8(model=weights, task="detect")
        return model
