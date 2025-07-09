from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Iterator, Optional
from unittest.mock import Mock, patch

import pytest
from cv2 import CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, VideoCapture
from numpy import ndarray

from OTVision.application.event.new_video_start import NewVideoStartEvent
from OTVision.detect.rtsp_input_source import convert_frame_to_rgb
from OTVision.domain.frame import Frame, FrameKeys
from OTVision.helpers.machine import ON_WINDOWS
from OTVision.plugin.ffmpeg_video_writer import (
    ConstantRateFactor,
    EncodingSpeed,
    FfmpegVideoWriter,
    PixelFormat,
    VideoCodec,
    VideoFormat,
    keep_original_save_location,
)
from tests.conftest import YieldFixture

FPS = 20


@pytest.mark.skipif(ON_WINDOWS, reason="Feature is not supported on Windows.")
class TestFfmpegVideoFileWriter:
    def test_write_video(
        self,
        target: FfmpegVideoWriter,
        cyclist_mp4: Path,
        save_location: Path,
        expected_save_location: Path,
    ) -> None:
        given = create_given_video(cyclist_mp4, save_location)
        target.open(
            given.save_location, width=given.width, height=given.height, fps=FPS
        )
        for frame in given.images:
            target.write(frame)
        target.close()

        given_frames = get_frames_from(cyclist_mp4)
        actual_frames = get_frames_from(expected_save_location)

        assert expected_save_location.exists()
        assert expected_save_location.stat().st_size > 0
        assert len(actual_frames) == len(given_frames)
        assert target.is_closed

    def test_open_and_close_writer(
        self, target: FfmpegVideoWriter, cyclist_mp4: Path, save_location: Path
    ) -> None:
        given = create_given_video(cyclist_mp4, save_location)

        assert target.is_closed
        target.open(
            given.save_location, width=given.width, height=given.height, fps=FPS
        )
        assert not target.is_closed
        target.close()
        assert target.is_closed

    def test_notify_on_flush_event(
        self, target: FfmpegVideoWriter, cyclist_mp4: Path, save_location: Path
    ) -> None:
        given = create_given_video(cyclist_mp4, save_location)

        target.open(
            given.save_location, width=given.width, height=given.height, fps=FPS
        )
        assert not target.is_closed
        target.notify_on_flush_event(Mock())
        assert target.is_closed

    @patch("OTVision.plugin.ffmpeg_video_writer.FfmpegVideoWriter.open")
    def test_notify_on_new_video_start(
        self,
        mock_open: Mock,
        target: FfmpegVideoWriter,
        cyclist_mp4: Path,
        save_location: Path,
    ) -> None:
        given_video = create_given_video(cyclist_mp4, save_location)
        given_event = derive_event_from(given_video)
        target.notify_on_new_video_start(given_event)

        mock_open.assert_called_once_with(
            given_event.output, given_event.width, given_event.height, given_event.fps
        )

    def test_filter(
        self,
        target: FfmpegVideoWriter,
        cyclist_mp4: Path,
        save_location: Path,
        expected_save_location: Path,
    ) -> None:
        given_video = create_given_video(cyclist_mp4, save_location)
        given_event = derive_event_from(given_video)
        given_frames = create_frames_from(given_video.images)

        target.notify_on_new_video_start(given_event)
        actual = list(target.filter(frame for frame in given_frames))

        assert actual == given_frames
        assert expected_save_location.exists()
        assert expected_save_location.stat().st_size > 0
        assert target.is_closed is False  # Writer should still be open

        # Close the writer to finalize the video file
        target.notify_on_flush_event(Mock())
        assert target.is_closed

        # Now that the writer is closed, we can read frames from the output file
        actual_frames = get_frames_from(expected_save_location)
        assert len(actual_frames) == len(given_frames)

    def test_filter_drops_frame_without_data(
        self,
        target: FfmpegVideoWriter,
        cyclist_mp4: Path,
        save_location: Path,
        expected_save_location: Path,
    ) -> None:
        given_video = create_given_video(cyclist_mp4, save_location)
        images = list(given_video.images)
        given_event = derive_event_from(given_video)
        # Ensure we have at least two valid frames (first and third)
        given_frames: list[Frame] = create_frames_from(
            iter([images[0], None, images[2]])
        )

        target.notify_on_new_video_start(given_event)
        actual = list(target.filter(frame for frame in given_frames))

        assert actual == given_frames
        assert expected_save_location.exists()
        assert target.is_closed is False  # Writer should still be open

        # Close the writer to finalize the video file
        target.notify_on_flush_event(Mock())
        assert target.is_closed

        # Now that the writer is closed, we can check the output file
        sleep(0.1)
        assert expected_save_location.stat().st_size > 0

        # We should have 2 frames in the output video (the first and third frames)
        actual_frames = get_frames_from(expected_save_location)
        # We expect 2 frames because one frame had None data and was skipped
        assert len(actual_frames) == 2


@dataclass
class GivenVideo:
    save_location: str
    images: Iterator[ndarray]
    width: int
    height: int


def create_given_video(video_file: Path, save_location: Path) -> GivenVideo:
    video_capture = VideoCapture(str(video_file))
    return GivenVideo(
        save_location=str(save_location),
        images=read_frames_from(video_capture),
        width=get_width(video_capture),
        height=get_height(video_capture),
    )


def derive_event_from(given_video: GivenVideo) -> NewVideoStartEvent:
    return NewVideoStartEvent(
        output=given_video.save_location,
        width=given_video.width,
        height=given_video.height,
        fps=FPS,
    )


def get_width(video_capture: VideoCapture) -> int:
    return int(video_capture.get(CAP_PROP_FRAME_WIDTH))


def get_height(video_capture: VideoCapture) -> int:
    return int(video_capture.get(CAP_PROP_FRAME_HEIGHT))


def get_frames_from(video_file: Path) -> list[ndarray]:
    sleep(0.1)
    video_capture = VideoCapture(str(video_file))
    if not video_capture.isOpened():
        video_capture.release()
        # The closing file in ffmpeg is started in a background thread.
        # We need to wait for it to finish.
        sleep(0.1)
        video_capture.open(str(video_file))

    return list(read_frames_from(video_capture))


def read_frames_from(video_capture: VideoCapture) -> Iterator[ndarray]:
    if not video_capture.isOpened():
        video_capture.release()

    while True:
        successful, frame = video_capture.read()
        if successful:
            yield convert_frame_to_rgb(frame)
        else:
            video_capture.release()
            break
    video_capture.release()


def create_frames_from(images: Iterator[Optional[ndarray]]) -> list[Frame]:
    frame_objects = []
    for i, frame_data in enumerate(images):
        frame_obj: Frame = {
            FrameKeys.data: frame_data,
            FrameKeys.frame: i,
            FrameKeys.source: str(cyclist_mp4),
            FrameKeys.output: str(save_location),
            FrameKeys.occurrence: datetime.now(),
        }
        frame_objects.append(frame_obj)
    return frame_objects


@pytest.fixture
def target() -> YieldFixture[FfmpegVideoWriter]:
    writer = FfmpegVideoWriter(
        save_location_strategy=keep_original_save_location,
        encoding_speed=EncodingSpeed.FAST,
        input_format=VideoFormat.RAW,
        output_format=VideoFormat.MP4,
        input_pixel_format=PixelFormat.RGB24,
        output_pixel_format=PixelFormat.YUV420P,
        output_video_codec=VideoCodec.H264_SOFTWARE,
        constant_rate_factor=ConstantRateFactor.DEFAULT,
    )
    yield writer
    # Ensure the writer is closed before the next test runs.
    writer.close()


@pytest.fixture
def cyclist_mp4(test_data_dir: Path) -> Path:
    return test_data_dir / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"


@pytest.fixture
def save_location(test_data_tmp_dir: Path, cyclist_mp4: Path) -> Path:
    return test_data_tmp_dir / cyclist_mp4.name


@pytest.fixture
def expected_save_location(save_location: Path) -> Path:
    return save_location
