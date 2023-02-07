import shutil
from pathlib import Path

import geopandas as gpd
import pytest
from pandas.testing import assert_frame_equal

from OTVision.transform.transform import main as transform


@pytest.fixture
def tmp_bzipped_ottrk_cyclist(test_data_dir: Path, test_data_tmp_dir: Path) -> Path:
    fname = "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.ottrk.bz2"
    ottrk_file = test_data_dir / fname
    dest = test_data_tmp_dir / Path(fname).stem
    shutil.copy2(ottrk_file, dest)
    return dest


@pytest.fixture
def tmp_bzipped_ottrk_truck(test_data_dir: Path, test_data_tmp_dir: Path) -> Path:
    fname = "Testvideo_Cars-Truck_FR20_2020-01-01_00-00-00.ottrk.bz2"
    ottrk_file = test_data_dir / fname
    dest = test_data_tmp_dir / Path(fname).stem
    shutil.copy2(ottrk_file, dest)
    return dest


@pytest.fixture
def single_refpts_file(test_data_dir: Path) -> Path:
    return test_data_dir / "Testvideo_FR20.otrfpts"


@pytest.fixture
def tmp_cyclist_refpts_file(test_data_dir: Path, test_data_tmp_dir: Path) -> Path:
    fname = "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.otrfpts"
    src = test_data_dir / fname
    dest = test_data_tmp_dir / fname
    shutil.copy2(src, dest)
    return dest


@pytest.fixture
def truck_gpkg(test_data_dir: Path) -> Path:
    return test_data_dir / "Testvideo_Cars-Truck_FR20_2020-01-01_00-00-00.gpkg"


@pytest.fixture
def cyclist_gpkg(test_data_dir: Path) -> Path:
    return test_data_dir / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.gpkg"


class TestTransform:
    """Tests the main function of OTVision/transform/transform.py
    transforming test tracks files from pixel to world coordinates based
    on a set of reference points in both pixel and world coordinates
    for both tracks file-specific reference points
    and uniform reference points for all test data.

    A tmp test directory is created and deleted by pytest fixture in test/conftest.py.
    """

    def test_transform_zippedOttrkFilesAs1stParam_correctRefptsFileAs2ndParam(
        self,
        single_refpts_file: Path,
        truck_gpkg: Path,
        cyclist_gpkg: Path,
        tmp_bzipped_ottrk_cyclist: Path,
        tmp_bzipped_ottrk_truck: Path,
    ) -> None:
        # sourcery skip: remove-assert-true, remove-redundant-pass

        transform(
            paths=[tmp_bzipped_ottrk_cyclist, tmp_bzipped_ottrk_truck],
            refpts_file=single_refpts_file,
        )

        # Compare gpkg files for all test data
        result_truck_gpkg_bzipped = tmp_bzipped_ottrk_truck.with_suffix(".gpkg")
        result_cyclist_gpkg_bzipped = tmp_bzipped_ottrk_cyclist.with_suffix(".gpkg")

        expected_utm_tracks_truck_df = gpd.read_file(truck_gpkg)
        result_utm_tracks_truck_df = gpd.read_file(
            result_truck_gpkg_bzipped, compression="bz2"
        )

        expected_utm_tracks_cyclist_df = gpd.read_file(cyclist_gpkg)
        result_utm_tracks_cyclist_df = gpd.read_file(
            result_cyclist_gpkg_bzipped, compression="bz2"
        )

        # Raise error if df's are not equal
        assert_frame_equal(expected_utm_tracks_truck_df, result_utm_tracks_truck_df)
        assert_frame_equal(expected_utm_tracks_cyclist_df, result_utm_tracks_cyclist_df)

    def test_transform_refptsFileInSameDir_zippedOttrkFilesAsParam(
        self,
        tmp_cyclist_refpts_file: Path,
        cyclist_gpkg: Path,
        tmp_bzipped_ottrk_cyclist: Path,
    ) -> None:
        # sourcery skip: remove-assert-true, remove-redundant-pass
        assert tmp_cyclist_refpts_file.exists()
        transform(paths=[tmp_bzipped_ottrk_cyclist])

        result_cyclist_gpkg_bzipped = tmp_bzipped_ottrk_cyclist.with_suffix(".gpkg")

        expected_utm_tracks_cyclist_df = gpd.read_file(cyclist_gpkg)
        result_utm_tracks_cyclist_df = gpd.read_file(
            result_cyclist_gpkg_bzipped, compression="bz2"
        )

        assert_frame_equal(expected_utm_tracks_cyclist_df, result_utm_tracks_cyclist_df)
