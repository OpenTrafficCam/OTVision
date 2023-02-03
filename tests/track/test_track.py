import shutil
from filecmp import cmpfiles
from pathlib import Path

import pytest

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files
from OTVision.track.track import main as track

SIGMA_L = CONFIG["TRACK"]["IOU"]["SIGMA_L"]
SIGMA_H = CONFIG["TRACK"]["IOU"]["SIGMA_H"]
SIGMA_IOU = CONFIG["TRACK"]["IOU"]["SIGMA_IOU"]
T_MIN = CONFIG["TRACK"]["IOU"]["T_MIN"]
T_MISS_MAX = CONFIG["TRACK"]["IOU"]["T_MISS_MAX"]


@pytest.mark.parametrize(
    "dir_name, sigma_l, sigma_h, sigma_iou, t_min, t_miss_max",
    [
        ("default", SIGMA_L, SIGMA_H, SIGMA_IOU, T_MIN, T_MISS_MAX),
        ("sigma_l_0_1", 0.1, SIGMA_H, SIGMA_IOU, T_MIN, T_MISS_MAX),
        ("sigma_l_0_5", 0.5, SIGMA_H, SIGMA_IOU, T_MIN, T_MISS_MAX),
        ("sigma_h_0_2", SIGMA_L, 0.2, SIGMA_IOU, T_MIN, T_MISS_MAX),
        ("sigma_h_0_6", SIGMA_L, 0.6, SIGMA_IOU, T_MIN, T_MISS_MAX),
        ("sigma_iou_0_2", SIGMA_L, SIGMA_H, 0.2, T_MIN, T_MISS_MAX),
        ("sigma_iou_0_6", SIGMA_L, SIGMA_H, 0.6, T_MIN, T_MISS_MAX),
        # ("t_min_3", SIGMA_L, SIGMA_H, SIGMA_IOU, 3, T_MISS_MAX),
        # ("t_min_10", SIGMA_L, SIGMA_H, SIGMA_IOU, 10, T_MISS_MAX),
        # ("t_miss_max_25", SIGMA_L, SIGMA_H, SIGMA_IOU, T_MIN, 25),
        # ("t_miss_max_75", SIGMA_L, SIGMA_H, SIGMA_IOU, T_MIN, 75),
    ],
)
def test_track_pass(
    test_data_tmp_dir: Path,
    test_data_dir: Path,
    dir_name: str,
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

    # Define sub dirs and create tmp folder
    test_track_dir = test_data_dir / "track" / dir_name
    (test_data_tmp_dir / "track").mkdir(exist_ok=True)
    test_track_tmp_dir = test_data_tmp_dir / "track" / dir_name
    test_track_tmp_dir.mkdir(exist_ok=True)

    # Get reference data
    ref_detections_files = get_files(
        paths=[test_track_dir],
        filetypes=[CONFIG["DEFAULT_FILETYPE"]["DETECT"]],
    )
    ref_tracks_files = get_files(
        paths=[test_track_dir], filetypes=[CONFIG["DEFAULT_FILETYPE"]["TRACK"]]
    )

    # Copy input data to temporary folder and start conversion
    for ref_detections_file in ref_detections_files:
        # test_detections_file = test_data_tmp_dir / "track" / ref_detections_file.name
        test_detections_file = test_track_tmp_dir / ref_detections_file.name
        shutil.copy2(
            ref_detections_file,
            test_detections_file,
        )

    # Track all test detections files
    dir_to_track = test_track_tmp_dir
    track(
        paths=[dir_to_track],
        sigma_l=sigma_l,
        sigma_h=sigma_h,
        sigma_iou=sigma_iou,
        t_min=t_min,
        t_miss_max=t_miss_max,
    )

    # Compare all test tracks files to their respective reference tracks files
    tracks_file_names = [file.name for file in ref_tracks_files]
    equal_files, different_files, irregular_files = cmpfiles(
        a=test_track_dir,
        b=test_track_tmp_dir,
        common=tracks_file_names,
        shallow=False,
    )
    assert tracks_file_names == equal_files
    assert not different_files
    assert not irregular_files
