from abc import ABC, abstractmethod
from typing import Iterator

from OTVision.application.config import DetectConfig
from OTVision.domain.frame import DetectedFrame, Frame


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
    def detect(self, frames: Iterator[Frame]) -> Iterator[DetectedFrame]:
        """Runs object detection on a video.

        Args:
            frames (Iterator[Frame]): the source to read frames from.

        Returns:
            Iterator[DetectedFrame]: nested list of detections.
                First level is frames, second level is detections within frame.
        """
        raise NotImplementedError

    @abstractmethod
    def preload(self) -> None:
        """Preload the model if possible."""
        raise NotImplementedError


class ObjectDetectorFactory(ABC):
    @abstractmethod
    def create(self, config: DetectConfig) -> ObjectDetector:
        raise NotImplementedError
