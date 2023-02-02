from pathlib import Path

import pytest


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
