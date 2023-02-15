from pathlib import Path

import pytest

from OTVision.detect.detect import main as detect


@pytest.fixture
def paths_with_legal_fileformats() -> list[Path]:
    return [
        Path("vid_a.mov"),
        Path("vid_b.mkv"),
        Path("vid_c.avi"),
        Path("vid_d.mpg"),
        Path("vid_e.mpeg"),
        Path("vid_f.m4v"),
        Path("vid_g.wmv"),
        Path("img_h.jpeg"),
        Path("img_i.jpg"),
        Path("img_j.png"),
    ]


@pytest.fixture
def paths_with_illegal_fileformats() -> list[Path]:
    return [Path("err_a.video"), Path("err_b.image")]


def test_detect_emptyDirAsParam(test_data_tmp_dir: Path) -> None:
    empty_dir = test_data_tmp_dir / "empty"
    empty_dir.mkdir()
    with pytest.raises(
        FileNotFoundError, match=r"No videos of type .* found to detect!"
    ):
        detect(paths=[empty_dir])


def test_detect_emptyListAsParam() -> None:
    with pytest.raises(
        FileNotFoundError, match=r"No videos of type .* found to detect!"
    ):
        detect(paths=[])
