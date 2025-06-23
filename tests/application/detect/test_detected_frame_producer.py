from unittest.mock import Mock

from OTVision.abstraction.pipes_and_filter import Filter
from OTVision.application.detect.detected_frame_producer import (
    SimpleDetectedFrameProducer,
)
from OTVision.domain.frame import DetectedFrame, Frame
from OTVision.domain.input_source_detect import InputSourceDetect

INPUT_SOURCE_GENERATOR = Mock()
VIDEO_FILE_WRITER_GENERATOR = Mock()
DETECTED_FRAME_GENERATOR = Mock()
DETECTED_FRAME_BUFFER_GENERATOR = Mock()


class TestSimpleDetectedFrameProducer:
    def test_produce(self) -> None:
        given_input_source = create_input_source()
        given_video_file_writer = create_video_file_writer()
        given_detection_filter = create_detection_filter()
        given_detected_frame_buffer = create_detected_frame_buffer()

        target = SimpleDetectedFrameProducer(
            given_input_source,
            given_video_file_writer,
            given_detection_filter,
            given_detected_frame_buffer,
        )
        actual = target.produce()

        assert actual == DETECTED_FRAME_BUFFER_GENERATOR
        given_input_source.produce.assert_called_once()
        given_video_file_writer.filter.assert_called_once_with(INPUT_SOURCE_GENERATOR)
        given_detection_filter.filter.assert_called_once_with(
            VIDEO_FILE_WRITER_GENERATOR
        )
        given_detected_frame_buffer.filter.assert_called_once_with(
            DETECTED_FRAME_GENERATOR
        )


def create_input_source() -> Mock:
    mock = Mock(spec=InputSourceDetect)
    mock.produce.return_value = INPUT_SOURCE_GENERATOR
    return mock


def create_detection_filter() -> Mock:
    mock = Mock(spec=Filter[Frame, DetectedFrame])
    mock.filter.return_value = DETECTED_FRAME_GENERATOR
    return mock


def create_detected_frame_buffer() -> Mock:
    mock = Mock(spec=Filter[Frame, DetectedFrame])
    mock.filter.return_value = DETECTED_FRAME_BUFFER_GENERATOR
    return mock


def create_video_file_writer() -> Mock:
    mock = Mock(spec=Filter[Frame, Frame])
    mock.filter.return_value = VIDEO_FILE_WRITER_GENERATOR
    return mock
