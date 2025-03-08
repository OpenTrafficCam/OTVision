from abc import ABC, abstractmethod
from typing import Generator

from OTVision.config import DetectConfig
from OTVision.domain.detection import DetectedFrame
from OTVision.domain.frame import Frame


class ObjectDetectorMetadata(ABC):
    @property
    @abstractmethod
    def config(self) -> DetectConfig:
        raise NotImplementedError

    @property
    @abstractmethod
    def classifications(self) -> dict[int, str]:
        """The model's classes that it is able to predict.

        Returns:
            dict[int, str]: the classes
        """
        raise NotImplementedError


class ObjectDetector(ObjectDetectorMetadata):

    @abstractmethod
    def detect(
        self, frames: Generator[Frame, None, None]
    ) -> Generator[DetectedFrame, None, None]:
        """Runs object detection on a video.

        Args:
            frames (Generator[Frame, None, None]): the source to read frames from.

        Returns:
            Generator[DetectedFrame, None, None]: nested list of detections.
                First level is frames, second level is detections within frame.
        """
        raise NotImplementedError


class ObjectDetectorFactory(ABC):
    @abstractmethod
    def create(self, config: DetectConfig) -> ObjectDetector:
        raise NotImplementedError
