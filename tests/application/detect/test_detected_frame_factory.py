from datetime import datetime
from unittest.mock import Mock

from OTVision.application.detect.detected_frame_factory import DetectedFrameFactory
from OTVision.domain.detection import DetectedFrame, Detection
from OTVision.domain.frame import Frame, FrameKeys


class TestDetectedFrameFactory:
    def test_create(self) -> None:
        given_frame = self.create_frame()
        given_detections = self.create_detections()

        target = DetectedFrameFactory()
        actual = target.create(given_frame, given_detections)

        assert actual == DetectedFrame(
            source=given_frame[FrameKeys.source],
            frame_number=given_frame[FrameKeys.frame],
            occurrence=given_frame[FrameKeys.occurrence],
            detections=given_detections,
        )

    def create_frame(self) -> Frame:
        return Frame(
            data=Mock(),
            frame=1,
            occurrence=datetime(2020, 1, 1, 12, 1, 1),
            source="path/to/source.mp4",
        )

    def create_detections(self) -> list[Detection]:
        return [Mock() for _ in range(3)]
