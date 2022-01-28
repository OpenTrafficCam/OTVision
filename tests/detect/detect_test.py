import pytest

from OTVision.detect.detect import _split_to_video_img_paths, FormatNotSupportedError


@pytest.fixture
def paths_with_legal_fileformats():
    return [
        "vid_a.mov",
        "vid_b.mkv",
        "vid_c.avi",
        "vid_d.mpg",
        "vid_e.mpeg",
        "vid_f.m4v",
        "vid_g.wmv",
        "img_h.jpeg",
        "img_i.jpg",
        "img_j.png",
    ]


@pytest.fixture
def paths_with_illegal_fileformats():
    return ["err_a.video", "err_b.image"]


def test_split_to_video_img_paths_legalPathsParam(paths_with_legal_fileformats):
    videos, imgs = _split_to_video_img_paths(paths_with_legal_fileformats)
    assert len(videos) == 7
    assert len(imgs) == 3


def test_split_to_video_img_paths_illegalPathsParam(paths_with_illegal_fileformats):
    with pytest.raises(FormatNotSupportedError):
        _split_to_video_img_paths(paths_with_illegal_fileformats)


def test_split_to_video_img_paths_emptyListsParam():
    videos, imgs = _split_to_video_img_paths([])
    assert not videos and not imgs
