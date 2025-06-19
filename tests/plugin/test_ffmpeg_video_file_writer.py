from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pytest
from cv2 import CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, VideoCapture
from numpy import ndarray

from OTVision.detect.rtsp_input_source import convert_frame_to_rgb
from OTVision.plugin.ffmpeg_video_writer import (
    EncodingSpeed,
    FfmpegVideoWriter,
    PixelFormat,
    VideoCodec,
    VideoFormat,
)

FPS = 20


class TestFfmpegVideoFileWriter:
    def test_write_video(self, cyclist_mp4: Path, save_location: Path) -> None:
        given = create_given_video(cyclist_mp4)
        target = FfmpegVideoWriter(
            encoding_speed=EncodingSpeed.FAST,
            input_format=VideoFormat.RAW,
            output_format=VideoFormat.MP4,
            input_pixel_format=PixelFormat.RGB,
            output_pixel_format=PixelFormat.YUV420P,
            output_video_codec=VideoCodec.H264,
            constant_rate_factor=23,
        )
        target.open(str(save_location), width=given.width, height=given.height, fps=FPS)
        for frame in given.frames:
            target.write(frame)
        target.close()

        given_frames = get_frames_from(cyclist_mp4)
        actual_frames = get_frames_from(save_location)

        assert save_location.exists()
        assert save_location.stat().st_size > 0
        assert len(actual_frames) == len(given_frames)


@dataclass
class GivenVideo:
    frames: Iterator[ndarray]
    width: int
    height: int


def create_given_video(video_file: Path) -> GivenVideo:
    video_capture = VideoCapture(str(video_file))
    return GivenVideo(
        frames=read_frames_from(video_capture),
        width=get_width(video_capture),
        height=get_height(video_capture),
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
        video_capture.release()
        raise StopIteration


@pytest.fixture
def cyclist_mp4(test_data_dir: Path) -> Path:
    return test_data_dir / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"


@pytest.fixture
def save_location(test_data_tmp_dir: Path, cyclist_mp4: Path) -> Path:
    return test_data_tmp_dir / cyclist_mp4.name
