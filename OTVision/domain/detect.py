from abc import ABC, abstractmethod
from typing import Generator, Self

from OTVision.config import DetectConfig
from OTVision.track.preprocess import Detection


class ObjectDetector(ABC):

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

    @abstractmethod
    def detect(self, source: str) -> Generator[list[Detection], None, None]:
        """Runs object detection on a video.

        Args:
            source (str): the source to read frames from.

        Returns:
            Generator[list[list[Detection], None, None]: nested list of detections.
                First level is frames, second level is detections within frame.
        """
        raise NotImplementedError

    @abstractmethod
    def configure_with(self, config: DetectConfig) -> Self:
        raise NotImplementedError
