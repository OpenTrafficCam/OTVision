import os
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from pandas.testing import assert_frame_equal

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files, remove_dir
from OTVision.helpers.machine import ON_WINDOWS
from OTVision.transform.transform import main as transform

TMP_DIR_NAME = "data_tmp"
TMP_DIR = Path(CONFIG["TESTDATAFOLDER"]).parent / TMP_DIR_NAME
UNIFORM_REFPTS_FILE = Path(CONFIG["TESTDATAFOLDER"]) / "Testvideo_FR20_uniform.otrfpts"


@pytest.mark.parametrize("uniform_refpts_file", [None, UNIFORM_REFPTS_FILE])
def test_transform(uniform_refpts_file):
    """Tests the main function of OTVision/transform/transform.py
    transforming test tracks files from pixel to world coordinates based
    on a set of reference points in both pixel and world coordinates
    for both tracks file-specific reference points
    and uniform reference points for all test data
    """

    # Get true ottrk and otrpfts files from tests/data
    true_ottrk_files = get_files(
        paths=Path(CONFIG["TESTDATAFOLDER"]),
        filetypes=".ottrk",
    )
    # Create tests/data_tmp
    if not TMP_DIR.is_dir():
        os.mkdir(TMP_DIR)
    for true_ottrk_file in true_ottrk_files:
        # Copy ottrk file to tests/data_tmp
        true_ottrk_file = Path(true_ottrk_file)
        test_ottrk_file = (
            true_ottrk_file.parents[1] / TMP_DIR_NAME / true_ottrk_file.name
        )
        shutil.copy2(
            true_ottrk_file,
            test_ottrk_file,
        )
        # Copy otrfpts file to tests/data_tmp
        true_otrfpts_file = Path(true_ottrk_file).with_suffix(".otrfpts")
        test_otrfpts_file = (
            true_otrfpts_file.parents[1] / TMP_DIR_NAME / true_otrfpts_file.name
        )
        shutil.copy2(
            true_otrfpts_file,
            test_otrfpts_file,
        )

    # Get test ottrk and otrpfts files from tests/data_tmp
    test_tracks_files = get_files(
        paths=Path(TMP_DIR),
        filetypes=".ottrk",
    )

    # Transform list of .ottrk files using otrefpts
    transform(tracks_files=test_tracks_files, uniform_refpts_file=uniform_refpts_file)

    # Compare gpkg files for all test data
    for true_ottrk_file in true_ottrk_files:
        # Get gpkg file names and read to df's
        true_gpkg_file = Path(true_ottrk_file).with_suffix(".gpkg")
        test_gpkg_file = true_gpkg_file.parents[1] / TMP_DIR_NAME / true_gpkg_file.name
        true_utm_tracks_df = gpd.read_file(true_gpkg_file)
        test_utm_tracks_df = gpd.read_file(test_gpkg_file)

        # Raise error if df's are not equal
        assert_frame_equal(true_utm_tracks_df, test_utm_tracks_df)

    # Remove tests/data_tmp and all contents
    remove_dir(TMP_DIR)

    assert True
