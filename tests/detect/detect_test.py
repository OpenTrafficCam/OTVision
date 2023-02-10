import bz2
import json
import shutil
from filecmp import cmp
from pathlib import Path

import pytest
import torch

import OTVision.config as config
from OTVision.detect.detect import main as detect
from tests.conftest import YieldFixture


@pytest.fixture
def detect_test_data_dir(test_data_dir: Path) -> Path:
    return test_data_dir / "detect"


@pytest.fixture
def detect_test_tmp_dir(test_data_tmp_dir: Path) -> YieldFixture[Path]:
    detect_tmp_dir = test_data_tmp_dir / "detect"
    detect_tmp_dir.mkdir()
    yield detect_tmp_dir
    shutil.rmtree(detect_tmp_dir)


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


@pytest.fixture
def cyclist_mp4(detect_test_data_dir: Path, detect_test_tmp_dir: Path) -> Path:
    fname = "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"
    src = detect_test_data_dir / fname
    dest = detect_test_tmp_dir / fname
    shutil.copy2(src, dest)
    return dest


@pytest.fixture
def truck_mp4(detect_test_data_dir: Path, detect_test_tmp_dir: Path) -> Path:
    fname = "Testvideo_Cars-Truck_FR20_2020-01-01_00-00-00.mp4"
    src = detect_test_data_dir / fname
    dest = detect_test_tmp_dir / fname
    shutil.copy2(src, dest)
    return dest


@pytest.fixture
def default_cyclist_otdet(detect_test_data_dir: Path) -> Path:
    fname = "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.otdet"
    return detect_test_data_dir / "default" / fname


class TestDetect:
    filetypes: list[str] = config.CONFIG[config.FILETYPES][config.VID]
    model = torch.hub.load(
        repo_or_dir="ultralytics/yolov5",
        model="yolov5s",
        pretrained=True,
        force_reload=False,
    )

    def test_detect_mp4AsParam_returnCorrectOtdetFile(
        self, detect_test_tmp_dir: Path, cyclist_mp4: Path, default_cyclist_otdet: Path
    ) -> None:
        detect([cyclist_mp4], force_reload_torch_hub_cache=True)
        result_otdet = detect_test_tmp_dir / default_cyclist_otdet.name
        assert cmp(result_otdet, default_cyclist_otdet)

    def test_detect_otdet_valid_json(
        self, detect_test_tmp_dir: Path, cyclist_mp4: Path
    ) -> None:
        detect([cyclist_mp4], model=self.model, force_reload_torch_hub_cache=False)
        result_otdet = detect_test_tmp_dir / f"{cyclist_mp4.stem}.otdet"
        assert result_otdet.exists()
        try:
            otdet_file = bz2.open(str(result_otdet), "r")
            json.load(otdet_file)
        finally:
            otdet_file.close()
