import os
import shutil
from functools import cached_property
from pathlib import Path
from unittest.mock import Mock

import pytest

from OTVision import version
from OTVision.application.config import TrackConfig, _TrackIouConfig
from OTVision.config import CONFIG
from OTVision.helpers.files import read_json
from OTVision.track.builder import TrackBuilder
from OTVision.track.id_generator import StrIdGenerator
from OTVision.track.track import OtvisionTrack
from tests.conftest import YieldFixture

TEST_RUN_ID = "test-run-id"

SIGMA_L = CONFIG["TRACK"]["IOU"]["SIGMA_L"]
SIGMA_H = CONFIG["TRACK"]["IOU"]["SIGMA_H"]
SIGMA_IOU = CONFIG["TRACK"]["IOU"]["SIGMA_IOU"]
T_MIN = CONFIG["TRACK"]["IOU"]["T_MIN"]
T_MISS_MAX = CONFIG["TRACK"]["IOU"]["T_MISS_MAX"]

version.otvision_version = Mock(return_value="ignored")
version.ottrack_version = Mock(return_value="ignored")
version.otdet_version = Mock(return_value="ignored")


@pytest.fixture
def test_track_dir(test_data_dir: Path) -> Path:
    return test_data_dir / "track"


@pytest.fixture
def test_track_tmp_dir(
    test_data_tmp_dir: Path, test_track_dir: Path
) -> YieldFixture[Path]:
    # Create tmp dir
    test_track_tmp_dir = test_data_tmp_dir / "track"
    test_track_tmp_dir.mkdir(exist_ok=True)
    # Copy test files to tmp dir
    shutil.copytree(test_track_dir, test_track_tmp_dir, dirs_exist_ok=True)
    # Delete tracks files from tmp dir (we create them during the tests)
    extension = CONFIG["DEFAULT_FILETYPE"]["TRACK"]
    tracks_files_to_delete = test_track_tmp_dir.rglob(f"*{extension}")
    for f in tracks_files_to_delete:
        f.unlink()
    # Yield tmp dir
    yield test_track_tmp_dir
    # Teardown tmp dir after use in test
    shutil.rmtree(test_track_tmp_dir)


@pytest.mark.parametrize(
    "test_case, sigma_l, sigma_h, sigma_iou, t_min, t_miss_max",
    [
        ("default", SIGMA_L, SIGMA_H, SIGMA_IOU, T_MIN, T_MISS_MAX),
        ("sigma_l_0_1", 0.1, SIGMA_H, SIGMA_IOU, T_MIN, T_MISS_MAX),
        ("sigma_l_0_5", 0.5, SIGMA_H, SIGMA_IOU, T_MIN, T_MISS_MAX),
        ("sigma_h_0_2", SIGMA_L, 0.2, SIGMA_IOU, T_MIN, T_MISS_MAX),
        ("sigma_h_0_6", SIGMA_L, 0.6, SIGMA_IOU, T_MIN, T_MISS_MAX),
        ("sigma_iou_0_2", SIGMA_L, SIGMA_H, 0.2, T_MIN, T_MISS_MAX),
        ("sigma_iou_0_6", SIGMA_L, SIGMA_H, 0.6, T_MIN, T_MISS_MAX),
        ("t_min_3", SIGMA_L, SIGMA_H, SIGMA_IOU, 3, T_MISS_MAX),
        ("t_min_10", SIGMA_L, SIGMA_H, SIGMA_IOU, 10, T_MISS_MAX),
        ("t_miss_max_25", SIGMA_L, SIGMA_H, SIGMA_IOU, T_MIN, 25),
        ("t_miss_max_75", SIGMA_L, SIGMA_H, SIGMA_IOU, T_MIN, 75),
    ],
)
@pytest.mark.skip(reason="Only used to update test data when tracking changes")
def test_update_test_data(
    test_track_dir: Path,
    test_track_tmp_dir: Path,
    test_case: str,
    sigma_l: float,
    sigma_h: float,
    sigma_iou: float,
    t_min: int,
    t_miss_max: int,
) -> None:
    # Track all test detections files
    input_folder = test_track_tmp_dir.relative_to(os.getcwd())
    otvision_track = create_otvision_track(
        paths=[input_folder / test_case],
        sigma_l=sigma_l,
        sigma_h=sigma_h,
        sigma_iou=sigma_iou,
        t_min=t_min,
        t_miss_max=t_miss_max,
        overwrite=True,
    )
    otvision_track.start()
    shutil.copytree(
        test_track_tmp_dir / test_case, test_track_dir / test_case, dirs_exist_ok=True
    )


@pytest.mark.parametrize(
    "test_case, sigma_l, sigma_h, sigma_iou, t_min, t_miss_max",
    [
        ("default", SIGMA_L, SIGMA_H, SIGMA_IOU, T_MIN, T_MISS_MAX),
        ("sigma_l_0_1", 0.1, SIGMA_H, SIGMA_IOU, T_MIN, T_MISS_MAX),
        ("sigma_l_0_5", 0.5, SIGMA_H, SIGMA_IOU, T_MIN, T_MISS_MAX),
        ("sigma_h_0_2", SIGMA_L, 0.2, SIGMA_IOU, T_MIN, T_MISS_MAX),
        ("sigma_h_0_6", SIGMA_L, 0.6, SIGMA_IOU, T_MIN, T_MISS_MAX),
        ("sigma_iou_0_2", SIGMA_L, SIGMA_H, 0.2, T_MIN, T_MISS_MAX),
        ("sigma_iou_0_6", SIGMA_L, SIGMA_H, 0.6, T_MIN, T_MISS_MAX),
        ("t_min_3", SIGMA_L, SIGMA_H, SIGMA_IOU, 3, T_MISS_MAX),
        ("t_min_10", SIGMA_L, SIGMA_H, SIGMA_IOU, 10, T_MISS_MAX),
        ("t_miss_max_25", SIGMA_L, SIGMA_H, SIGMA_IOU, T_MIN, 25),
        ("t_miss_max_75", SIGMA_L, SIGMA_H, SIGMA_IOU, T_MIN, 75),
    ],
)
def test_track_pass(
    test_track_dir: Path,
    test_track_tmp_dir: Path,
    test_case: str,
    sigma_l: float,
    sigma_h: float,
    sigma_iou: float,
    t_min: int,
    t_miss_max: int,
) -> None:
    """Tests the main function of OTVision/track/track.py
    tracking detections from .otdet files (from short test videos)
    using the IOU tracker by Bochinski et al.
    """

    # Track all test detections files
    input_folder = test_track_tmp_dir.relative_to(os.getcwd())
    otvision_track = create_otvision_track(
        paths=[input_folder / test_case],
        sigma_l=sigma_l,
        sigma_h=sigma_h,
        sigma_iou=sigma_iou,
        t_min=t_min,
        t_miss_max=t_miss_max,
        overwrite=True,
    )
    otvision_track.start()

    # Get reference tracks file names
    extension = CONFIG["DEFAULT_FILETYPE"]["TRACK"]
    ref_tracks_files = (test_track_dir / test_case).glob(f"*{extension}")
    tracks_file_names = [file.name for file in ref_tracks_files]

    # Compare all test tracks files to their respective reference tracks files
    equal_files = []
    different_files = []
    irregular_files = []
    for file in tracks_file_names:
        a = test_track_dir / test_case / file
        b = test_track_tmp_dir / test_case / file
        try:
            json_a = read_json(a)
            json_b = read_json(b)
            if json_a == json_b:
                equal_files.append(file)
            else:
                different_files.append(file)
        except Exception:
            irregular_files.append(file)
    for equal_file in equal_files:
        assert equal_file in tracks_file_names
    for different_file in different_files:
        a = test_track_dir / test_case / different_file
        b = test_track_tmp_dir / test_case / different_file
        json_a = read_json(a)
        json_b = read_json(b)
        assert json_b == json_a
    assert not different_files
    assert not irregular_files


@pytest.mark.parametrize("overwrite", [(True), (False)])
def test_track_overwrite(test_track_tmp_dir: Path, overwrite: bool) -> None:
    """Tests if the main function of OTVision/track/track.py properly overwrites
    existing files or not based on the overwrite parameter"""

    # Get tracks files to test for
    test_case = "default"
    extension = CONFIG["DEFAULT_FILETYPE"]["TRACK"]
    test_tracks_files = (test_track_tmp_dir / test_case).glob(f"*{extension}")

    # Track all test detections files for a first time and get file statistics
    otvision_track = create_otvision_track(
        paths=[test_track_tmp_dir / test_case], overwrite=overwrite
    )
    otvision_track.start()
    pre_test_file_stats = [file.stat().st_mtime_ns for file in test_tracks_files]

    # Track all test detections files for a second time and get file statistics
    otvision_track = create_otvision_track(
        paths=[test_track_tmp_dir / test_case], overwrite=overwrite
    )
    otvision_track.start()
    post_test_file_stats = [file.stat().st_mtime_ns for file in test_tracks_files]

    # Check if file statistics are different
    for pre, post in zip(pre_test_file_stats, post_test_file_stats):
        if overwrite:
            assert pre != post
        else:
            assert pre == post


@pytest.mark.parametrize(
    "paths, message",
    [
        (1, "Paths needs to be a sequence"),
        ("some_str", "some_str is neither a file nor a dir"),
        (Path("some_str"), "Paths needs to be a sequence"),
        (
            [Path("some_str"), Path("some_other_str")],
            "some_str is neither a file nor a dir",
        ),
    ],
)
def test_track_fail_wrong_paths(paths, message) -> None:  # type: ignore
    """Tests if the main function of OTVision/track/track.py raises errors when wrong
    paths are given"""

    with pytest.raises(ValueError, match=message):
        otvision_track = create_otvision_track(paths=paths)
        otvision_track.start()


@pytest.mark.parametrize(
    "sigma_l, sigma_h, sigma_iou, t_min, t_miss_max",
    [
        ("some_str", SIGMA_H, SIGMA_IOU, T_MIN, T_MISS_MAX),
        (Path("some_str"), SIGMA_H, SIGMA_IOU, T_MIN, T_MISS_MAX),
        (SIGMA_L, "some_str", SIGMA_IOU, T_MIN, T_MISS_MAX),
        (SIGMA_L, Path("some_str"), SIGMA_IOU, T_MIN, T_MISS_MAX),
        (SIGMA_L, SIGMA_H, "some_str", T_MIN, T_MISS_MAX),
        (SIGMA_L, SIGMA_H, Path("some_str"), T_MIN, T_MISS_MAX),
        (SIGMA_L, SIGMA_H, SIGMA_IOU, "some_str", T_MISS_MAX),
        (SIGMA_L, SIGMA_H, SIGMA_IOU, Path("some_str"), T_MISS_MAX),
        (SIGMA_L, SIGMA_H, SIGMA_IOU, 10.5, T_MISS_MAX),
        (SIGMA_L, SIGMA_H, SIGMA_IOU, T_MIN, "some_str"),
        (SIGMA_L, SIGMA_H, SIGMA_IOU, T_MIN, Path("some_str")),
        (SIGMA_L, SIGMA_H, SIGMA_IOU, T_MIN, 75.5),
    ],
)
def test_track_fail_wrong_parameters(
    test_track_tmp_dir: Path,
    sigma_l: float,
    sigma_h: float,
    sigma_iou: float,
    t_min: int,
    t_miss_max: int,
) -> None:
    """Tests if the main function of OTVision/track/track.py raises errors when wrong
    parameters are given"""

    test_case = "default"

    # Track all test detections files
    with pytest.raises(ValueError, match=".*has to be.*"):
        otvision_track = create_otvision_track(
            paths=[test_track_tmp_dir / test_case],
            sigma_l=sigma_l,
            sigma_h=sigma_h,
            sigma_iou=sigma_iou,
            t_min=t_min,
            t_miss_max=t_miss_max,
        )
        otvision_track.start()


def test_track_emptyDirAsParam(test_track_tmp_dir: Path) -> None:
    empty_dir = test_track_tmp_dir / "empty"
    empty_dir.mkdir()

    otvision_track = create_otvision_track(paths=[empty_dir])
    otvision_track.start()

    assert os.listdir(empty_dir) == []


def create_otvision_track(
    paths: list,
    sigma_l: float = TrackConfig.iou.sigma_l,
    sigma_h: float = TrackConfig.iou.sigma_h,
    sigma_iou: float = TrackConfig.iou.sigma_iou,
    t_min: int = TrackConfig.iou.t_min,
    t_miss_max: int = TrackConfig.iou.t_miss_max,
    overwrite: bool = TrackConfig.overwrite,
) -> OtvisionTrack:
    track_config = create_track_config(
        paths, sigma_l, sigma_h, sigma_iou, t_min, t_miss_max, overwrite
    )
    builder = MockTrackBuilder()
    builder.update_current_track_config.update(track_config)
    return builder.build()


def create_track_config(
    paths: list[str],
    sigma_l: float,
    sigma_h: float,
    sigma_iou: float,
    t_min: int,
    t_miss_max: int,
    overwrite: bool = False,
) -> TrackConfig:
    return TrackConfig(
        paths=paths,
        run_chained=TrackConfig.run_chained,
        iou=_TrackIouConfig(
            sigma_l=sigma_l,
            sigma_h=sigma_h,
            sigma_iou=sigma_iou,
            t_min=t_min,
            t_miss_max=t_miss_max,
        ),
        overwrite=overwrite,
    )


class MockTrackBuilder(TrackBuilder):
    @cached_property
    def tracking_run_id_generator(self) -> StrIdGenerator:
        return lambda: TEST_RUN_ID
