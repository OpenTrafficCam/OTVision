from OTVision.application.detect.current_object_detector import CurrentObjectDetector
from OTVision.domain.object_detection import ObjectDetectorMetadata


class CurrentObjectDetectorMetadata:
    """Provider for metadata about the currently configured object detector.

    This class serves as a wrapper to access metadata information about the
    currently configured object detector.

    Args:
        current_object_detector (CurrentObjectDetector): The current object detector
            instance from which to retrieve metadata.
    """

    def __init__(self, current_object_detector: CurrentObjectDetector) -> None:
        self.current_object_detector = current_object_detector

    def get(self) -> ObjectDetectorMetadata:
        """Retrieve metadata about the currently configured object detector.

        Returns:
            ObjectDetectorMetadata: Metadata information about the current object
                detector configuration.
        """

        return self.current_object_detector.get()
