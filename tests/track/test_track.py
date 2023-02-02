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

    # Get reference data
    ref_detections_files = get_files(
        paths=[test_data_dir], filetypes=[CONFIG["DEFAULT_FILETYPE"]["DETECT"]]
    )
    ref_tracks_files = get_files(
        paths=[test_data_dir], filetypes=[CONFIG["DEFAULT_FILETYPE"]["TRACK"]]
    )

    # Copy input data to temporary folder and start conversion
    for ref_detections_file in ref_detections_files:
        shutil.copy2(
            Path(ref_detections_file),
            Path(ref_detections_file).parents[1]
            / test_data_tmp_dir.name
            / Path(ref_detections_file).name,
        )
    test_detections_file = get_files(
        paths=[test_data_tmp_dir],
        filetypes=[CONFIG["DEFAULT_FILETYPE"]["DETECT"]],
    )

    # Track all test detections files
    track(test_detections_file)

    # Compare all test tracks files to their respective reference tracks files
    tracks_file_names = [file.name for file in ref_tracks_files]
    equal_files, different_files, irregular_files = cmpfiles(
        a=test_data_dir, b=test_data_tmp_dir, common=tracks_file_names, shallow=False
    )
    assert tracks_file_names == equal_files
