import os
import shutil
from filecmp import cmpfiles
from pathlib import Path
from unittest.mock import Mock

import pytest

from OTVision import version
from OTVision.config import CONFIG
from OTVision.track.track import main as track
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
    track(
        paths=[input_folder / test_case],
        sigma_l=sigma_l,
        sigma_h=sigma_h,
        sigma_iou=sigma_iou,
        t_min=t_min,
        t_miss_max=t_miss_max,
        tracking_run_id_generator=lambda: TEST_RUN_ID,
    )
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
    track(
        paths=[input_folder / test_case],
        sigma_l=sigma_l,
        sigma_h=sigma_h,
        sigma_iou=sigma_iou,
        t_min=t_min,
        t_miss_max=t_miss_max,
        tracking_run_id_generator=lambda: TEST_RUN_ID,
    )

    # Get reference tracks file names
    extension = CONFIG["DEFAULT_FILETYPE"]["TRACK"]
    ref_tracks_files = (test_track_dir / test_case).glob(f"*{extension}")
    tracks_file_names = [file.name for file in ref_tracks_files]

    # Compare all test tracks files to their respective reference tracks files
    equal_files, different_files, irregular_files = cmpfiles(
        a=test_track_dir / test_case,
        b=test_track_tmp_dir / test_case,
        common=tracks_file_names,
        shallow=False,
    )
    for equal_file in equal_files:
        assert equal_file in tracks_file_names
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
    track(paths=[test_track_tmp_dir / test_case])
    pre_test_file_stats = [file.stat().st_mtime_ns for file in test_tracks_files]

    # Track all test detections files for a second time and get file statistics
    track(paths=[test_track_tmp_dir / test_case], overwrite=overwrite)
    post_test_file_stats = [file.stat().st_mtime_ns for file in test_tracks_files]

    # Check if file statistics are different
    for pre, post in zip(pre_test_file_stats, post_test_file_stats):
        if overwrite:
            assert pre != post
        else:
            assert pre == post


@pytest.mark.parametrize(
    "paths",
    [
        (1),
        ("some_str"),
        (Path("some_str")),
        ([Path("some_str"), Path("some_other_str")]),
    ],
)
def test_track_fail_wrong_paths(paths) -> None:  # type: ignore
    """Tests if the main function of OTVision/track/track.py raises errors when wrong
    paths are given"""

    # Check if TypeError is raised
    with pytest.raises(TypeError, match=r"Paths needs to be a list of pathlib.Path"):
        track(paths=paths)


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
        track(
            paths=[test_track_tmp_dir / test_case],
            sigma_l=sigma_l,
            sigma_h=sigma_h,
            sigma_iou=sigma_iou,
            t_min=t_min,
            t_miss_max=t_miss_max,
        )


def test_track_emptyDirAsParam(test_track_tmp_dir: Path) -> None:
    empty_dir = test_track_tmp_dir / "empty"
    empty_dir.mkdir()

    track(paths=[empty_dir])

    assert os.listdir(empty_dir) == []
