from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock, call, patch

import pytest
from cv2 import CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH

from OTVision.application.config import DATETIME_FORMAT, StreamConfig
from OTVision.detect.detected_frame_buffer import FlushEvent
from OTVision.detect.rtsp_input_source import Counter, RtspInputSource
from OTVision.domain.frame import Frame

START_TIME = datetime(2020, 1, 1, 12, 0, 0)
INIT_TIME = START_TIME - timedelta(seconds=1)
FIRST_OCCURRENCE = START_TIME + timedelta(seconds=1)
SECOND_OCCURRENCE = FIRST_OCCURRENCE + timedelta(seconds=2)
THIRD_OCCURRENCE = SECOND_OCCURRENCE + timedelta(seconds=1)
FOURTH_OCCURRENCE = THIRD_OCCURRENCE + timedelta(seconds=1)

WIDTH = 800
HEIGHT = 600
OUTPUT_FPS = 1.0

FIRST_FRAME_DATA = Mock()
SECOND_FRAME_DATA = Mock()
THIRD_FRAME_DATA = Mock()
FOURTH_FRAME_DATA = Mock()
FIFTH_FRAME_DATA = Mock()

FIRST_FRAME_RGB_DATA = Mock()
THIRD_FRAME_RGB_DATA = Mock()
FOURTH_FRAME_RGB_DATA = Mock()
FIFTH_FRAME_RGB_DATA = Mock()
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
FIRST_OUTPUT = str(
    STREAM_CONFIG.save_dir / f"{STREAM_NAME}_FR{round(OUTPUT_FPS)}"
    f"_{FIRST_OCCURRENCE.strftime(DATETIME_FORMAT)}.mp4"
)
THIRD_OUTPUT = str(
    STREAM_CONFIG.save_dir / f"{STREAM_NAME}_FR{round(OUTPUT_FPS)}"
    f"_{FIRST_OCCURRENCE.strftime(DATETIME_FORMAT)}.mp4"
)
FOURTH_OUTPUT = str(
    STREAM_CONFIG.save_dir / f"{STREAM_NAME}_FR{round(OUTPUT_FPS)}"
    f"_{THIRD_OCCURRENCE.strftime(DATETIME_FORMAT)}.mp4"
)
FIFTH_OUTPUT = str(
    STREAM_CONFIG.save_dir / f"{STREAM_NAME}_FR{round(OUTPUT_FPS)}"
    f"_{THIRD_OCCURRENCE.strftime(DATETIME_FORMAT)}.mp4"
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
                output=FIRST_OUTPUT,
                occurrence=FIRST_OCCURRENCE,
            ),
            Frame(
                data=THIRD_FRAME_RGB_DATA,
                frame=2,
                source=RTSP_URL,
                output=THIRD_OUTPUT,
                occurrence=SECOND_OCCURRENCE,
            ),
            Frame(
                data=FOURTH_FRAME_RGB_DATA,
                frame=1,
                source=RTSP_URL,
                output=FOURTH_OUTPUT,
                occurrence=THIRD_OCCURRENCE,
            ),
            Frame(
                data=FIFTH_FRAME_RGB_DATA,
                frame=2,
                source=RTSP_URL,
                output=FIFTH_OUTPUT,
                occurrence=FOURTH_OCCURRENCE,
            ),
        ]
        assert given.datetime_provider.provide.call_count == 6
        assert given.video_capture_instance.read.call_count == 5
        assert given.convert_frame_to_rgb.call_args_list == [
            call(FIRST_FRAME_DATA),
            call(THIRD_FRAME_DATA),
            call(FOURTH_FRAME_DATA),
            call(FIFTH_FRAME_DATA),
        ]
        assert given.video_capture_instance.get.call_args_list == [
            call(CAP_PROP_FRAME_WIDTH),
            call(CAP_PROP_FRAME_HEIGHT),
            call(CAP_PROP_FRAME_WIDTH),
            call(CAP_PROP_FRAME_HEIGHT),
            call(CAP_PROP_FRAME_WIDTH),
            call(CAP_PROP_FRAME_HEIGHT),
        ]
        assert given.subject.notify.call_args_list == create_expected_flush_events()


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
        (True, FOURTH_FRAME_DATA),
        (True, FIFTH_FRAME_DATA),
    ]

    given.video_capture_instance.isOpened.return_value = video_capture_is_opened
    given.video_capture_instance.read.side_effect = frames
    given.video_capture_instance.get.side_effect = [
        WIDTH,
        HEIGHT,
        WIDTH,
        HEIGHT,
        WIDTH,
        HEIGHT,
    ]

    given.datetime_provider.provide.side_effect = [
        INIT_TIME,
        START_TIME,
        FIRST_OCCURRENCE,
        SECOND_OCCURRENCE,
        THIRD_OCCURRENCE,
        FOURTH_OCCURRENCE,
    ]
    given.video_capture.return_value = given.video_capture_instance

    given.convert_frame_to_rgb.side_effect = [
        FIRST_FRAME_RGB_DATA,
        THIRD_FRAME_RGB_DATA,
        FOURTH_FRAME_RGB_DATA,
        FIFTH_FRAME_RGB_DATA,
    ]
    given.config.stream = STREAM_CONFIG
    given.config.convert.output_fps = OUTPUT_FPS
    given.get_current_config.get.return_value = given.config

    return RtspInputSource(
        subject=given.subject,
        datetime_provider=given.datetime_provider,
        frame_counter=given.frame_counter,
        get_current_config=given.get_current_config,
    )


def create_expected_flush_events() -> list[Any]:
    return [
        call(create_expected_flush_event(FIRST_OCCURRENCE)),
        call(create_expected_flush_event(THIRD_OCCURRENCE)),
        call(create_expected_flush_event(THIRD_OCCURRENCE)),
    ]


def create_expected_flush_event(start_time: datetime) -> FlushEvent:
    return FlushEvent.create(
        source=RTSP_URL,
        output=str(
            STREAM_SAVE_DIR / f"{STREAM_NAME}_FR{round(OUTPUT_FPS)}"
            f"_{start_time.strftime(DATETIME_FORMAT)}.mp4"
        ),
        duration=timedelta(seconds=2),
        source_width=WIDTH,
        source_height=HEIGHT,
        source_fps=OUTPUT_FPS,
        start_time=start_time,
    )
