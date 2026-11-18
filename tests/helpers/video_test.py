from datetime import timedelta
from pathlib import Path

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


def test_get_video_dimensions(cyclist_mp4: Path) -> None:
    assert list(get_video_dimensions(cyclist_mp4)) == [800, 600]


def test_get_fps(cyclist_mp4: Path) -> None:
    assert get_fps(cyclist_mp4) == 20.0


def test_get_duration(cyclist_mp4: Path) -> None:
    assert get_duration(cyclist_mp4) == timedelta(seconds=3)


def test_get_number_of_frames(cyclist_mp4: Path) -> None:
    assert get_number_of_frames(cyclist_mp4) == 60
