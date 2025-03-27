from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest
from cv2 import CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH

from OTVision.application.config import DATETIME_FORMAT, StreamConfig
from OTVision.detect.detected_frame_buffer import FlushEvent
from OTVision.detect.rtsp_input_source import Counter, RtspInputSource
from OTVision.domain.frame import Frame

START_TIME = datetime(2020, 1, 1, 12, 0, 0)
FIRST_OCCURRENCE = START_TIME + timedelta(seconds=1)
SECOND_OCCURRENCE = FIRST_OCCURRENCE + timedelta(seconds=2)

WIDTH = 800
HEIGHT = 600
OUTPUT_FPS = 1

FIRST_FRAME_DATA = Mock()
SECOND_FRAME_DATA = Mock()
THIRD_FRAME_DATA = Mock()

FIRST_FRAME_RGB_DATA = Mock()
SECOND_FRAME_RGB_DATA = Mock()
RTSP_URL = "rtsp://192.168.1.100:554/1/h264preview"
STREAM_NAME = "OTCamera15"
STREAM_SAVE_DIR = Path("path/to/save/dir")
FLUSH_BUFFER_SIZE = 2
STREAM_CONFIG = StreamConfig(
    name=STREAM_NAME,
    source=RTSP_URL,
    save_dir=STREAM_SAVE_DIR,
    flush_buffer_size=FLUSH_BUFFER_SIZE,
)


@dataclass
class Given:
    subject: Mock
    datetime_provider: Mock
    frame_counter: Counter
    flush_buffer_size: int
    get_current_config: Mock
    video_capture: Mock
    video_capture_instance: Mock
    convert_frame_to_rgb: Mock
    config: Mock


class TestRtspInputSource:
    @patch("OTVision.detect.rtsp_input_source.convert_frame_to_rgb")
    @patch("OTVision.detect.rtsp_input_source.VideoCapture")
    def test_produce(
        self, mock_video_capture: Mock, mock_convert_frame_to_rgb: Mock
    ) -> None:
        given = create_given(mock_video_capture, mock_convert_frame_to_rgb)
        target = create_target_with(given)
        generator = target.produce()
        actual = list()
        actual.append(next(generator))
        actual.append(next(generator))
        target.stop()
        with pytest.raises(StopIteration):
            next(generator)

        assert actual == [
            Frame(
                data=FIRST_FRAME_RGB_DATA,
                frame=1,
                source=RTSP_URL,
                occurrence=FIRST_OCCURRENCE,
            ),
            Frame(
                data=SECOND_FRAME_RGB_DATA,
                frame=2,
                source=RTSP_URL,
                occurrence=SECOND_OCCURRENCE,
            ),
        ]
        assert given.datetime_provider.provide.call_count == 3
        assert given.video_capture_instance.read.call_count == 3
        assert given.convert_frame_to_rgb.call_args_list == [
            call(FIRST_FRAME_DATA),
            call(THIRD_FRAME_DATA),
        ]
        assert given.video_capture_instance.get.call_args_list == [
            call(CAP_PROP_FRAME_WIDTH),
            call(CAP_PROP_FRAME_HEIGHT),
            call(CAP_PROP_FRAME_WIDTH),
            call(CAP_PROP_FRAME_HEIGHT),
        ]
        expected_flush_event = FlushEvent.create(
            source=RTSP_URL,
            output=str(
                STREAM_SAVE_DIR / f"{STREAM_NAME}_FR{OUTPUT_FPS}"
                f"_{START_TIME.strftime(DATETIME_FORMAT)}.mp4"
            ),
            duration=timedelta(seconds=2),
            source_width=WIDTH,
            source_height=HEIGHT,
            source_fps=OUTPUT_FPS,
            start_time=START_TIME,
        )
        assert given.subject.notify.call_args_list == [
            call(expected_flush_event),
            call(expected_flush_event),
        ]


def create_given(video_capture: Mock, convert_frame_to_rgb: Mock) -> Given:
    return Given(
        subject=Mock(),
        datetime_provider=Mock(),
        frame_counter=Counter(),
        flush_buffer_size=FLUSH_BUFFER_SIZE,
        get_current_config=Mock(),
        video_capture=video_capture,
        video_capture_instance=Mock(),
        convert_frame_to_rgb=convert_frame_to_rgb,
        config=Mock(),
    )


def create_target_with(
    given: Given, video_capture_is_opened: bool = True
) -> RtspInputSource:
    frames = [
        (True, FIRST_FRAME_DATA),
        (False, SECOND_FRAME_DATA),
        (True, THIRD_FRAME_DATA),
    ]

    given.video_capture_instance.isOpened.return_value = video_capture_is_opened
    given.video_capture_instance.read.side_effect = frames
    given.video_capture_instance.get.side_effect = [WIDTH, HEIGHT, WIDTH, HEIGHT]

    given.datetime_provider.provide.side_effect = [
        START_TIME,
        FIRST_OCCURRENCE,
        SECOND_OCCURRENCE,
    ]
    given.video_capture.return_value = given.video_capture_instance

    given.convert_frame_to_rgb.side_effect = [
        FIRST_FRAME_RGB_DATA,
        SECOND_FRAME_RGB_DATA,
    ]
    given.config.stream = STREAM_CONFIG
    given.config.convert.output_fps = OUTPUT_FPS
    given.get_current_config.get.return_value = given.config

    return RtspInputSource(
        subject=given.subject,
        datetime_provider=given.datetime_provider,
        frame_counter=given.frame_counter,
        flush_buffer_size=given.flush_buffer_size,
        get_current_config=given.get_current_config,
    )
