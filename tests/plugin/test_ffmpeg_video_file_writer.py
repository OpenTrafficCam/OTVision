from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator
from unittest.mock import Mock, patch

import pytest
from cv2 import CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, VideoCapture
from numpy import ndarray

from OTVision.application.event.new_video_start import NewVideoStartEvent
from OTVision.detect.rtsp_input_source import convert_frame_to_rgb
from OTVision.domain.frame import Frame, FrameKeys
from OTVision.plugin.ffmpeg_video_writer import (
    ConstantRateFactor,
    EncodingSpeed,
    FfmpegVideoWriter,
    PixelFormat,
    VideoCodec,
    VideoFormat,
)

FPS = 20


class TestFfmpegVideoFileWriter:
    def test_write_video(self, cyclist_mp4: Path, save_location: Path) -> None:
        given = create_given_video(cyclist_mp4, save_location)
        target = create_target()
        target.open(
            given.save_location, width=given.width, height=given.height, fps=FPS
        )
        for frame in given.images:
            target.write(frame)
        target.close()

        given_frames = get_frames_from(cyclist_mp4)
        actual_frames = get_frames_from(save_location)

        assert save_location.exists()
        assert save_location.stat().st_size > 0
        assert len(actual_frames) == len(given_frames)
        assert target.is_closed

    def test_open_and_close_writer(
        self, cyclist_mp4: Path, save_location: Path
    ) -> None:
        given = create_given_video(cyclist_mp4, save_location)
        target = create_target()

        assert target.is_closed
        target.open(
            given.save_location, width=given.width, height=given.height, fps=FPS
        )
        assert not target.is_closed
        target.close()
        assert target.is_closed

    def test_notify_on_flush_event(
        self, cyclist_mp4: Path, save_location: Path
    ) -> None:
        given = create_given_video(cyclist_mp4, save_location)
        target = create_target()

        target.open(
            given.save_location, width=given.width, height=given.height, fps=FPS
        )
        assert not target.is_closed
        target.notify_on_flush_event(Mock())
        assert target.is_closed

    @patch("OTVision.plugin.ffmpeg_video_writer.FfmpegVideoWriter.open")
    def test_notify_on_new_video_start(
        self, mock_open: Mock, cyclist_mp4: Path, save_location: Path
    ) -> None:
        given_video = create_given_video(cyclist_mp4, save_location)
        given_event = derive_event_from(given_video)
        target = create_target()
        target.notify_on_new_video_start(given_event)

        mock_open.assert_called_once_with(
            given_event.output, given_event.width, given_event.height, given_event.fps
        )

    def test_filter(self, cyclist_mp4: Path, save_location: Path) -> None:
        given_video = create_given_video(cyclist_mp4, save_location)
        given_event = derive_event_from(given_video)
        given_frames = create_frames_from(given_video.images)
        target = create_target()

        target.notify_on_new_video_start(given_event)
        actual = list(target.filter(frame for frame in given_frames))

        assert actual == given_frames

        assert save_location.exists()
        assert save_location.stat().st_size > 0
        assert target.is_closed is False  # Writer should still be open

        # Close the writer
        target.notify_on_flush_event(Mock())
        assert target.is_closed


def create_target() -> FfmpegVideoWriter:
    return FfmpegVideoWriter(
        encoding_speed=EncodingSpeed.FAST,
        input_format=VideoFormat.RAW,
        output_format=VideoFormat.MP4,
        input_pixel_format=PixelFormat.RGB24,
        output_pixel_format=PixelFormat.YUV420P,
        output_video_codec=VideoCodec.H264,
        constant_rate_factor=ConstantRateFactor.LOSSLESS,
    )


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
    video_capture = VideoCapture(str(video_file))
    return list(read_frames_from(video_capture))


def read_frames_from(video_capture: VideoCapture) -> Iterator[ndarray]:
    if not video_capture.isOpened():
        video_capture.release()
        raise ValueError("Cannot open the video file/stream.")

    while True:
        successful, frame = video_capture.read()
        if successful:
            yield convert_frame_to_rgb(frame)
        else:
            video_capture.release()
            break


def create_frames_from(images: Iterator[ndarray]) -> list[Frame]:
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
def cyclist_mp4(test_data_dir: Path) -> Path:
    return test_data_dir / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"


@pytest.fixture
def save_location(test_data_tmp_dir: Path, cyclist_mp4: Path) -> Path:
    return test_data_tmp_dir / cyclist_mp4.name
