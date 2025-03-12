from typing import Generator

from OTVision.abstraction.pipes_and_filter import Filter
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.domain.detection import DetectedFrame
from OTVision.domain.frame import Frame
from OTVision.domain.object_detection import ObjectDetector, ObjectDetectorFactory


class CurrentObjectDetector(Filter[Frame, DetectedFrame]):
    """Use case to retrieve the currently used object detector.

    Filter implementation for detecting objects in frames using current
    configuration.

    Args:
        get_current_config (GetCurrentConfig): Provider of current configuration.
        factory (ObjectDetectorFactory): Factory for creating object detector instances.
    """

    def __init__(
        self, get_current_config: GetCurrentConfig, factory: ObjectDetectorFactory
    ) -> None:
        self._get_current_config = get_current_config
        self._factory = factory

    def get(self) -> ObjectDetector:
        """Retrieve the currently used object detector.

        Returns:
            ObjectDetector: The object detector.

        """
        detect_config = self._get_current_config.get().detect
        return self._factory.create(detect_config)

    def filter(
        self, pipe: Generator[Frame, None, None]
    ) -> Generator[DetectedFrame, None, None]:
        return self.get().detect(pipe)
