from pathlib import Path

import pytest

from OTVision.detect.pyav_frame_count_provider import PyAVFrameCountProvider


@pytest.fixture
def cyclist_mp4(test_data_dir: Path) -> Path:
    return (
        test_data_dir / "detect" / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"
    )


class TestPyAvFrameCountProvider:
    def test_provide(self, cyclist_mp4: Path) -> None:
        target = PyAVFrameCountProvider()
        actual = target.provide(cyclist_mp4)
        assert actual == 60
