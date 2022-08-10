import shutil
from pathlib import Path

import geopandas as gpd
import pytest
from pandas.testing import assert_frame_equal

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files
from OTVision.transform.transform import main as transform

SINGLE_REFPTS_FILE = Path(CONFIG["TESTDATAFOLDER"]) / "Testvideo_FR20.otrfpts"


@pytest.mark.parametrize("single_refpts_file", [None, SINGLE_REFPTS_FILE])
def test_transform(test_data_tmp_dir: Path, single_refpts_file):
    # sourcery skip: remove-assert-true, remove-redundant-pass
    """Tests the main function of OTVision/transform/transform.py
    transforming test tracks files from pixel to world coordinates based
    on a set of reference points in both pixel and world coordinates
    for both tracks file-specific reference points
    and uniform reference points for all test data.

    A tmp test directory is created and deleted by pytest fixture in test/conftest.py.
    """

    # Get true ottrk and otrpfts files from tests/data
    true_ottrk_files = get_files(
        paths=Path(CONFIG["TESTDATAFOLDER"]),
        filetypes=".ottrk",
    )
    for true_ottrk_file in true_ottrk_files:
        # Copy ottrk file to tests/data_tmp
        true_ottrk_file = Path(true_ottrk_file)
        test_ottrk_file = (
            true_ottrk_file.parents[1] / test_data_tmp_dir.name / true_ottrk_file.name
        )
        shutil.copy2(
            true_ottrk_file,
            test_ottrk_file,
        )
        # Copy otrfpts file to tests/data_tmp
        true_otrfpts_file = Path(true_ottrk_file).with_suffix(".otrfpts")
        test_otrfpts_file = (
            true_otrfpts_file.parents[1]
            / test_data_tmp_dir.name
            / true_otrfpts_file.name
        )
        shutil.copy2(
            true_otrfpts_file,
            test_otrfpts_file,
        )

    # Get test ottrk and otrpfts files from tests/data_tmp
    test_tracks_files = get_files(
        paths=Path(test_data_tmp_dir),
        filetypes=".ottrk",
    )

    # Transform list of .ottrk files using otrefpts
    transform(paths=test_tracks_files, single_refpts_file=single_refpts_file)

    # Compare gpkg files for all test data
    for true_ottrk_file in true_ottrk_files:
        # Get gpkg file names and read to df's
        true_gpkg_file = Path(true_ottrk_file).with_suffix(".gpkg")
        test_gpkg_file = (
            true_gpkg_file.parents[1] / test_data_tmp_dir.name / true_gpkg_file.name
        )
        true_utm_tracks_df = gpd.read_file(true_gpkg_file)
        test_utm_tracks_df = gpd.read_file(test_gpkg_file)

        # Raise error if df's are not equal
        assert_frame_equal(true_utm_tracks_df, test_utm_tracks_df)

    assert True
