from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from OTVision.abstraction.observer import AsyncSubject
from OTVision.detect.detected_frame_buffer import (
    DetectedFrameBuffer,
    DetectedFrameBufferEvent,
    FlushEvent,
)
from OTVision.domain.detection import Detection
from OTVision.domain.frame import DetectedFrame
from tests.utils.mocking import create_mocks


class TestDetectedFrameBuffer:
    @pytest.fixture
    def subject_mock(self) -> AsyncMock:
        return AsyncMock(spec=AsyncSubject)

    @pytest.fixture
    def target(self, subject_mock: AsyncMock) -> DetectedFrameBuffer:
        return DetectedFrameBuffer(subject=subject_mock)

    @pytest.mark.asyncio
    async def test_on_flush_notifies_subject_with_buffer_event(
        self, target: DetectedFrameBuffer, subject_mock: AsyncMock
    ) -> None:
        frames: list[DetectedFrame] = create_mocks(2)
        source_metadata = Mock()
        flush_event = FlushEvent(source_metadata=source_metadata)

        await target._notify_observers(frames, flush_event)

        expected_event: DetectedFrameBufferEvent = DetectedFrameBufferEvent(
            source_metadata=source_metadata, frames=frames
        )
        subject_mock.notify.assert_called_once_with(expected_event)
        assert target._get_buffered_elements() == []

    @pytest.mark.asyncio
    async def test_frames_are_buffered_without_image_data(
        self, target: DetectedFrameBuffer
    ) -> None:
        """
        #Bug https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7623
        """  # noqa
        frame_number = 1
        occurrence = datetime(2020, 1, 1, 12, 0, 0)
        source = "my_source"
        output = "path/to/output.mp4"
        detections: list[Detection] = create_mocks(3)
        image = Mock()

        given_frame = DetectedFrame(
            no=frame_number,
            occurrence=occurrence,
            source=source,
            output=output,
            detections=detections,
            image=image,
        )
        await target.buffer(given_frame)
        actual = target._get_buffered_elements()[0]

        expected = DetectedFrame(
            no=frame_number,
            occurrence=occurrence,
            source=source,
            output=output,
            detections=detections,
            image=None,
        )

        assert actual == expected
