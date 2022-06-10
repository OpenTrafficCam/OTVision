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

TMP_FOLDER = "data_tmp"


def test_transform():
    """Tests the main function of OTVision/transform/transform.py
    transforming tracks file from pixel to world coordinates based
    on a set of reference points in both pixel and world coordinates
    """

    # Get ottrk and otrpfts files from tests/data

    ref_tracks_files = get_files(
        paths=Path(CONFIG["TESTDATAFOLDER"]),
        filetypes=".ottrk",
    )
    # Create tests/data_tmp
    test_data_tmp_dir = Path(CONFIG["TESTDATAFOLDER"]).parent / TMP_FOLDER
    if not test_data_tmp_dir.is_dir():
        os.mkdir(test_data_tmp_dir)
    for true_ottrk_file in ref_tracks_files:
        # Copy ottrk file to tests/data_tmp
        true_ottrk_file = Path(true_ottrk_file)
        test_ottrk_file = true_ottrk_file.parents[1] / TMP_FOLDER / true_ottrk_file.name
        shutil.copy2(
            true_ottrk_file,
            test_ottrk_file,
        )
        # Copy otrfpts file to tests/data_tmp
        true_otrfpts_file = Path(true_ottrk_file).with_suffix(".otrfpts")
        test_otrfpts_file = (
            true_otrfpts_file.parents[1] / TMP_FOLDER / true_otrfpts_file.name
        )
        shutil.copy2(
            true_otrfpts_file,
            test_otrfpts_file,
        )
        # Transform ottrk using otrefpts
        transform(tracks_files=test_ottrk_file, refpts_file=test_otrfpts_file)

        # Get gpkg file names and read to df's
        true_gpkg_file = Path(true_ottrk_file).with_suffix(".gpkg")
        test_gpkg_file = true_gpkg_file.parents[1] / TMP_FOLDER / true_gpkg_file.name
        true_utm_tracks_df = gpd.read_file(true_gpkg_file)
        test_utm_tracks_df = gpd.read_file(test_gpkg_file)

        # Raise error if df's are not equal
        assert_frame_equal(true_utm_tracks_df, test_utm_tracks_df)

    # Remove tests/data_tmp and all contents
    remove_dir(test_data_tmp_dir)

    assert True
