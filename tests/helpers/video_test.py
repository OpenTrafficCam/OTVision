from datetime import timedelta
from pathlib import Path

import av
import pytest

from OTVision.helpers.video import (
    get_duration,
    get_fps,
    get_number_of_frames,
    get_video_dimensions,
)


@pytest.fixture
def cyclist_mp4(test_data_dir: Path) -> Path:
    return test_data_dir / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"


@pytest.fixture
def rotated_cyclist_mp4(test_data_dir: Path) -> Path:
    """Cyclist video with a display rotation of 180 degrees."""
    return (
        test_data_dir
        / "detect"
        / "rotated-Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"
    )


@pytest.fixture
def rotated_by_90_degrees_cyclist_mp4(cyclist_mp4: Path, tmp_path: Path) -> Path:
    """Copy of the cyclist video with a display rotation of 90 degrees."""
    rotated_video = tmp_path / cyclist_mp4.name
    with (
        av.open(str(cyclist_mp4)) as source,
        av.open(str(rotated_video), "w") as target,
    ):
        source_stream = source.streams.video[0]
        target_stream = target.add_stream_from_template(source_stream)
        target_stream.set_display_rotation(90)
        for packet in source.demux(source_stream):
            if packet.dts is None:
                continue
            packet.stream = target_stream
            target.mux(packet)
    return rotated_video


def test_get_video_dimensions(cyclist_mp4: Path) -> None:
    assert list(get_video_dimensions(cyclist_mp4)) == [800, 600]


def test_get_fps(cyclist_mp4: Path) -> None:
    assert get_fps(cyclist_mp4) == 20.0


def test_get_duration(cyclist_mp4: Path) -> None:
    assert get_duration(cyclist_mp4) == timedelta(seconds=3)


def test_get_number_of_frames(cyclist_mp4: Path) -> None:
    assert get_number_of_frames(cyclist_mp4) == 60


def test_rotated_video_by_180_degrees_keeps_dimensions(
    rotated_cyclist_mp4: Path,
) -> None:
    assert list(get_video_dimensions(rotated_cyclist_mp4)) == [800, 600]
    assert get_fps(rotated_cyclist_mp4) == 20.0
    assert get_duration(rotated_cyclist_mp4) == timedelta(seconds=3)
    assert get_number_of_frames(rotated_cyclist_mp4) == 60


def test_rotated_video_by_90_degrees_swaps_dimensions(
    rotated_by_90_degrees_cyclist_mp4: Path,
) -> None:
    assert list(get_video_dimensions(rotated_by_90_degrees_cyclist_mp4)) == [600, 800]
