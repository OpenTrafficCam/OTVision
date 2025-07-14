from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock, call, patch

import pytest

from OTVision.application.config import DATETIME_FORMAT, StreamConfig
from OTVision.application.event.new_video_start import NewVideoStartEvent
from OTVision.detect.detected_frame_buffer import FlushEvent
from OTVision.detect.rtsp_input_source import Counter, RtspInputSource
from OTVision.domain.frame import Frame

RTSP_INPUT_SOURCE_MODULE = "OTVision.detect.rtsp_input_source"
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
    subject_flush_event: Mock
    subject_new_video_start: Mock
    datetime_provider: Mock
    frame_counter: Counter
    flush_buffer_size: int
    get_current_config: Mock
    video_capture: Mock
    video_capture_instance: Mock
    convert_frame_to_rgb: Mock
    config: Mock


class TestRtspInputSource:
    @patch(RTSP_INPUT_SOURCE_MODULE + ".is_connection_available", return_value=True)
    @patch(RTSP_INPUT_SOURCE_MODULE + ".convert_frame_to_rgb")
    @patch(RTSP_INPUT_SOURCE_MODULE + ".VideoCapture")
    def test_produce(
        self,
        mock_video_capture: Mock,
        mock_convert_frame_to_rgb: Mock,
        mock_is_connection_available: Mock,
    ) -> None:
        given = setup_with(create_given(mock_video_capture, mock_convert_frame_to_rgb))
        target = create_target(given)
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
        assert (
            given.subject_flush_event.notify.call_args_list
            == create_expected_flush_events()
        )
        assert given.subject_new_video_start.notify.call_args_list == [
            call(create_expected_new_video_start(FIRST_OUTPUT)),
            call(create_expected_new_video_start(FOURTH_OUTPUT)),
        ]
        mock_is_connection_available.assert_called_once_with(RTSP_URL)

    @patch(RTSP_INPUT_SOURCE_MODULE + ".is_connection_available", return_value=True)
    @patch(RTSP_INPUT_SOURCE_MODULE + ".convert_frame_to_rgb")
    @patch(RTSP_INPUT_SOURCE_MODULE + ".VideoCapture")
    def test_reconnecting_on_consecutive_read_fails(
        self,
        mock_video_capture: Mock,
        mock_convert_frame_to_rgb: Mock,
        mock_is_connection_available: Mock,
    ) -> None:
        first_vc_instance = Mock()
        second_vc_instance = Mock()
        third_vc_instance = Mock()

        given = create_given(mock_video_capture, mock_convert_frame_to_rgb)
        given = setup_with(given, video_capture_is_opened=False)

        # This triggers the reconnecting
        first_vc_instance.read.side_effect = [
            (True, FIRST_FRAME_DATA),
            (False, None),
            (False, None),
        ]

        # This simulates the first re-connect try that fails
        second_vc_instance.isOpened.return_value = False

        # This simulates the second re-connect try that succeeds
        third_vc_instance.read.side_effect = [(True, SECOND_FRAME_DATA)]
        third_vc_instance.isOpened.return_value = True

        given.video_capture.side_effect = [
            first_vc_instance,
            second_vc_instance,
            third_vc_instance,
        ]

        target = create_target(given)
        target._read_fail_threshold = 2

        actual = list()
        actual.append(target._read_next_frame())  # First read successful
        actual.append(target._read_next_frame())  # Second read fails
        actual.append(target._read_next_frame())  # Third read fails
        actual.append(target._read_next_frame())  # Fourth read successful

        assert actual == [FIRST_FRAME_DATA, None, None, SECOND_FRAME_DATA]
        first_vc_instance.release.assert_called_once()
        assert first_vc_instance.read.call_count == 3
        second_vc_instance.isOpened.assert_called_once()
        second_vc_instance.read.assert_not_called()
        assert second_vc_instance.read.call_count == 0

        third_vc_instance.release.assert_not_called()
        assert third_vc_instance.read.call_count == 1

        assert target._consecutive_read_fails == 0
        assert mock_is_connection_available.call_args_list == [
            call(RTSP_URL),
            call(RTSP_URL),
            call(RTSP_URL),
        ]


def create_given(video_capture: Mock, convert_frame_to_rgb: Mock) -> Given:
    return Given(
        subject_flush_event=Mock(),
        subject_new_video_start=Mock(),
        datetime_provider=Mock(),
        frame_counter=Counter(),
        flush_buffer_size=FLUSH_BUFFER_SIZE,
        get_current_config=Mock(),
        video_capture=video_capture,
        video_capture_instance=Mock(),
        convert_frame_to_rgb=convert_frame_to_rgb,
        config=Mock(),
    )


def create_target(given: Given) -> RtspInputSource:
    return RtspInputSource(
        subject_flush=given.subject_flush_event,
        subject_new_video_start=given.subject_new_video_start,
        datetime_provider=given.datetime_provider,
        frame_counter=given.frame_counter,
        get_current_config=given.get_current_config,
    )


def setup_with(given: Given, video_capture_is_opened: bool = True) -> Given:
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

    return given


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


def create_expected_new_video_start(output: str) -> NewVideoStartEvent:
    return NewVideoStartEvent(output=output, width=WIDTH, height=HEIGHT, fps=OUTPUT_FPS)
