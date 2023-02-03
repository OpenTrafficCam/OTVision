import shutil
from filecmp import cmpfiles
from pathlib import Path

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files
from OTVision.track.track import main as track


def test_track_pass(test_data_tmp_dir: Path, test_data_dir: Path) -> None:
    """Tests the main function of OTVision/track/track.py
    tracking detections from .otdet files (from short test videos)
    using the IOU tracker by Bochinski et al.
    """

    # Define sub dirs and create tmp folder
    test_tracks_dir = test_data_dir / "track"
    test_tracks_tmp_dir = test_data_tmp_dir / "track"
    test_tracks_tmp_dir.mkdir(exist_ok=True)

    # Get reference data
    ref_detections_files = get_files(
        paths=[test_tracks_dir],
        filetypes=[CONFIG["DEFAULT_FILETYPE"]["DETECT"]],
    )
    ref_tracks_files = get_files(
        paths=[test_tracks_dir], filetypes=[CONFIG["DEFAULT_FILETYPE"]["TRACK"]]
    )

    # Copy input data to temporary folder and start conversion
    for ref_detections_file in ref_detections_files:
        # test_detections_file = test_data_tmp_dir / "track" / ref_detections_file.name
        test_detections_file = test_tracks_tmp_dir / ref_detections_file.name
        shutil.copy2(
            ref_detections_file,
            test_detections_file,
        )

    # Track all test detections files
    dir_to_track = test_tracks_tmp_dir
    track(paths=[dir_to_track])

    # Compare all test tracks files to their respective reference tracks files
    tracks_file_names = [file.name for file in ref_tracks_files]
    equal_files, different_files, irregular_files = cmpfiles(
        a=test_tracks_dir,
        b=test_tracks_tmp_dir,
        common=tracks_file_names,
        shallow=False,
    )
    assert tracks_file_names == equal_files
    assert not different_files
    assert not irregular_files
