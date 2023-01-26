from pathlib import Path

import pytest

from OTVision.detect.detect import FormatNotSupportedError, _split_to_video_img_paths


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


def test_split_to_video_img_paths_legalPathsParam(
    paths_with_legal_fileformats: list[Path],
) -> None:
    videos, imgs = _split_to_video_img_paths(paths_with_legal_fileformats)
    assert len(videos) == 7
    assert len(imgs) == 3


def test_split_to_video_img_paths_illegalPathsParam(
    paths_with_illegal_fileformats: list[Path],
) -> None:
    with pytest.raises(FormatNotSupportedError):
        _split_to_video_img_paths(paths_with_illegal_fileformats)


def test_split_to_video_img_paths_emptyListsParam() -> None:
    videos, imgs = _split_to_video_img_paths([])
    assert not videos and not imgs
