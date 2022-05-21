import os
import shutil
from pathlib import Path

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files, remove_dir
from OTVision.helpers.machine import ON_WINDOWS
from OTVision.transform.transform import main as transform


def test_convert():
    """Tests the main function of OTVision/transform/transform.py
    transforming tracks file from pixel to world coordinates based
    on a set of reference points in both pixel and world coordinates
    """

    try:
        # test_data_tmp_dir = Path(CONFIG["TESTDATAFOLDER"]).parent / "data_tmp"

        tracks_files = get_files(
            paths=Path(CONFIG["TESTDATAFOLDER"]) / "Testvideo_FR20_Cars-Cyclist.ottrk",
            filetypes="ottrk",
        )
        refpts_file = get_files(
            paths=Path(CONFIG["TESTDATAFOLDER"]) / "Testvideo_FR20_Cars-Cyclist.otrfpts",
            filetypes="otrfpts",
        )
        transform(tracks_files=tracks_files, refpts_file=refpts_file)

        # Remove test data tmp dir
        # remove_dir(test_data_tmp_dir)

        # TODO: Compare with reference data

    except Exception as e:
        assert False, e
