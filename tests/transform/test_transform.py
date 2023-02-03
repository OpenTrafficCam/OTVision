import shutil
from filecmp import cmpfiles
from pathlib import Path
from typing import Union

import geopandas as gpd
import pytest
from pandas.testing import assert_frame_equal

from OTVision.helpers.files import get_files
from OTVision.transform.transform import main as transform


@pytest.mark.parametrize("single_refpts_file_name", [None, "Testvideo_FR20.otrfpts"])
def test_transform(
    test_data_tmp_dir: Path,
    test_data_dir: Path,
    single_refpts_file_name: Union[None, str],
) -> None:
    # sourcery skip: remove-assert-true, remove-redundant-pass
    """Tests the main function of OTVision/transform/transform.py
    transforming test tracks files from pixel to world coordinates based
    on a set of reference points in both pixel and world coordinates
    for both tracks file-specific reference points
    and uniform reference points for all test data.

    A tmp test directory is created and deleted by pytest fixture in test/conftest.py.
    """

    # Define sub dirs and create tmp folder
    test_transform_dir = test_data_dir / "transform"
    (test_data_tmp_dir / "transform").mkdir(exist_ok=True)
    test_transform_tmp_dir = test_data_tmp_dir / "transform"
    test_transform_tmp_dir.mkdir(exist_ok=True)

    # Get true ottrk and otrpfts files from tests/data
    ref_ottrk_files = get_files(
        paths=[test_transform_dir],
        filetypes=[".ottrk"],
    )

    ref_gpkg_files = get_files(
        paths=[test_transform_tmp_dir],
        filetypes=[".ottrk"],
    )
    for ref_ottrk_file in ref_ottrk_files:
        # Copy input data to temporary folder and start conversion
        test_ottrk_file = test_transform_tmp_dir / ref_ottrk_file.name
        shutil.copy2(
            ref_ottrk_file,
            test_ottrk_file,
        )
        if not single_refpts_file_name:
            ref_otrfpts_file = ref_ottrk_file.with_suffix(".otrfpts")
            test_otrfpts_file = test_transform_tmp_dir / ref_otrfpts_file.name
            shutil.copy2(
                ref_otrfpts_file,
                test_otrfpts_file,
            )
    if single_refpts_file_name:
        ref_otrfpts_file = test_transform_dir / single_refpts_file_name
        test_otrfpts_file = test_transform_tmp_dir / single_refpts_file_name
        shutil.copy2(
            ref_otrfpts_file,
            test_otrfpts_file,
        )

    # Get test ottrk and otrpfts files from tests/data_tmp
    test_ottrk_files = get_files(
        paths=[test_data_tmp_dir],
        filetypes=[".ottrk"],
    )

    # Transform list of .ottrk files using otrefpts
    if single_refpts_file_name:
        transform(
            paths=test_ottrk_files, refpts_file=test_data_dir / single_refpts_file_name
        )
    else:
        transform(paths=test_ottrk_files, refpts_file=None)

    # Compare all test tracks files to their respective reference tracks files
    gpkg_file_names = [file.name for file in ref_ottrk_files]
    equal_file_names, different_file_names, irregular_file_names = cmpfiles(
        a=test_transform_dir,
        b=test_transform_tmp_dir,
        common=gpkg_file_names,
        shallow=False,
    )
    assert gpkg_file_names == equal_file_names
    assert not different_file_names
    assert not irregular_file_names

    # Compare gpkg files for all test data
    # for ref_ottrk_file in ref_ottrk_files:
    #     # Get gpkg file names and read to df's
    #     true_gpkg_file = Path(ref_ottrk_file).with_suffix(".gpkg")
    #     test_gpkg_file = (
    #         true_gpkg_file.parents[1] / test_data_tmp_dir.name / true_gpkg_file.name
    #     )
    #     true_utm_tracks_df = gpd.read_file(true_gpkg_file)
    #     test_utm_tracks_df = gpd.read_file(test_gpkg_file)

    #     # Raise error if df's are not equal
    #     assert_frame_equal(true_utm_tracks_df, test_utm_tracks_df)

    # assert True
