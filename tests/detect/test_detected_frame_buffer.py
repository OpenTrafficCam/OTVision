from unittest.mock import Mock

import pytest

from OTVision.abstraction.observer import Subject
from OTVision.detect.detected_frame_buffer import (
    DetectedFrameBuffer,
    DetectedFrameBufferEvent,
    FlushEvent,
)
from OTVision.domain.detection import DetectedFrame
from tests.utils.mocking import create_mocks


class TestDetectedFrameBuffer:
    @pytest.fixture
    def subject_mock(self) -> Mock:
        return Mock(spec=Subject)

    @pytest.fixture
    def target(self, subject_mock: Mock) -> DetectedFrameBuffer:
        return DetectedFrameBuffer(subject=subject_mock)

    def test_on_flush_notifies_subject_with_buffer_event(
        self, target: DetectedFrameBuffer, subject_mock: Mock
    ) -> None:
        frames: list[DetectedFrame] = create_mocks(2)
        source_metadata = Mock()
        flush_event = FlushEvent(source_metadata=source_metadata)

        target._notify_observers(frames, flush_event)

        expected_event: DetectedFrameBufferEvent = DetectedFrameBufferEvent(
            source_metadata=source_metadata, frames=frames
        )
        subject_mock.notify.assert_called_once_with(expected_event)
        assert target._get_buffered_elements() == []
