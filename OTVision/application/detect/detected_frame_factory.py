from OTVision.domain.detection import DetectedFrame, Detection
from OTVision.domain.frame import Frame, FrameKeys


class DetectedFrameFactory:
    """Factory for creating DetectedFrame objects from Frame and Detection data."""

    def create(self, frame: Frame, detections: list[Detection]) -> DetectedFrame:
        """Creates a DetectedFrame object from a Frame and its detections.

        Args:
            frame (Frame): the frame object.
            detections (list[Detection]): The detections to be associated with
                the frame.

        Returns:
            DetectedFrame: A new DetectedFrame instance containing the combined frame
                and detection information.
        """

        return DetectedFrame(
            source=frame[FrameKeys.source],
            frame_number=frame[FrameKeys.frame],
            occurrence=frame[FrameKeys.occurrence],
            detections=detections,
        )
