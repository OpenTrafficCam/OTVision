from unittest.mock import Mock

import pytest

from OTVision.abstraction.pipes_and_filter import Filter
from OTVision.application.config import Config, DetectConfig
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.detect.detected_frame_producer_factory import DetectedFrameProducerFactory
from OTVision.domain.frame import DetectedFrame, Frame
from OTVision.domain.input_source_detect import InputSourceDetect

INPUT_SOURCE_GENERATOR = Mock()
VIDEO_FILE_WRITER_GENERATOR = Mock()
DETECTED_FRAME_GENERATOR = Mock()
DETECTED_FRAME_BUFFER_GENERATOR = Mock()

PRODUCER_WITH_VIDEO_WRITER = Mock()
PRODUCER_WITHOUT_VIDEO_WRITER = Mock()


class TestDetectedFrameProducerFactory:
    @pytest.mark.parametrize("write_video", [False])
    def test_create_with_video_writer(self, write_video: bool) -> None:
        given_input_source = create_input_source()
        given_video_file_writer = create_video_file_writer()
        given_detection_filter = create_detection_filter()
        given_detected_frame_buffer = create_detected_frame_buffer()
        given_get_current_config = create_get_current_config(write_video=write_video)

        target = DetectedFrameProducerFactory(
            input_source=given_input_source,
            video_writer_filter=given_video_file_writer,
            detection_filter=given_detection_filter,
            detected_frame_buffer=given_detected_frame_buffer,
            get_current_config=given_get_current_config,
        )
        producer = target.create()

        assert producer == DETECTED_FRAME_BUFFER_GENERATOR
        given_input_source.produce.assert_called_once()
        if write_video:
            given_video_file_writer.filter.assert_called_once_with(
                INPUT_SOURCE_GENERATOR
            )
            given_detection_filter.filter.assert_called_once_with(
                VIDEO_FILE_WRITER_GENERATOR
            )
        else:
            given_video_file_writer.filter.assert_not_called()
            given_detection_filter.filter.assert_called_once_with(
                INPUT_SOURCE_GENERATOR
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


def create_get_current_config(write_video: bool) -> Mock:
    mock = Mock(spec=GetCurrentConfig)
    mock.get.return_value = Config(detect=DetectConfig(write_video=write_video))
    return mock
