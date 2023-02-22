import shutil
from pathlib import Path

import geopandas as gpd
import pytest
from pandas.testing import assert_frame_equal

from OTVision.transform.transform import main as transform
from tests.conftest import YieldFixture


@pytest.fixture
def test_data_transform_dir(test_data_dir: Path) -> Path:
    return test_data_dir / "transform"


@pytest.fixture
def test_data_tmp_transform_dir(
    test_data_tmp_dir: Path,
) -> YieldFixture[Path]:
    data_tmp_transform_dir = test_data_tmp_dir / "transform"
    data_tmp_transform_dir.mkdir(exist_ok=True)
    yield data_tmp_transform_dir
    shutil.rmtree(data_tmp_transform_dir)


@pytest.fixture
def tmp_ottrk_cyclist_bz2(
    test_data_transform_dir: Path, test_data_tmp_transform_dir: Path
) -> Path:
    fname = "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.ottrk"
    ottrk_file = test_data_transform_dir / fname
    dest = test_data_tmp_transform_dir / fname
    shutil.copy2(ottrk_file, dest)
    return dest


@pytest.fixture
def tmp_ottrk_truck_bz2(
    test_data_transform_dir: Path, test_data_tmp_transform_dir: Path
) -> Path:
    fname = "Testvideo_Cars-Truck_FR20_2020-01-01_00-00-00.ottrk"
    ottrk_file = test_data_transform_dir / fname
    dest = test_data_tmp_transform_dir / fname
    shutil.copy2(ottrk_file, dest)
    return dest


@pytest.fixture
def single_refpts_file(test_data_transform_dir: Path) -> Path:
    return test_data_transform_dir / "Testvideo_FR20.otrfpts"


@pytest.fixture
def tmp_cyclist_refpts_file(
    test_data_transform_dir: Path, test_data_tmp_transform_dir: Path
) -> YieldFixture[Path]:
    fname = "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.otrfpts"
    src = test_data_transform_dir / fname
    dest = test_data_tmp_transform_dir / fname
    shutil.copy2(src, dest)
    yield dest
    dest.unlink()


@pytest.fixture
def truck_gpkg(test_data_transform_dir: Path) -> Path:
    return (
        test_data_transform_dir / "Testvideo_Cars-Truck_FR20_2020-01-01_00-00-00.gpkg"
    )


@pytest.fixture
def cyclist_gpkg(test_data_transform_dir: Path) -> Path:
    return (
        test_data_transform_dir / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.gpkg"
    )


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
        tmp_ottrk_cyclist_bz2: Path,
        tmp_ottrk_truck_bz2: Path,
    ) -> None:
        # sourcery skip: remove-assert-true, remove-redundant-pass

        transform(
            paths=[tmp_ottrk_cyclist_bz2, tmp_ottrk_truck_bz2],
            refpts_file=single_refpts_file,
        )

        # Compare gpkg files for all test data
        result_truck_gpkg = tmp_ottrk_truck_bz2.with_suffix(".gpkg")
        result_cyclist_gpkg = tmp_ottrk_cyclist_bz2.with_suffix(".gpkg")

        expected_utm_tracks_truck_df = gpd.read_file(truck_gpkg)
        result_utm_tracks_truck_df = gpd.read_file(result_truck_gpkg)

        expected_utm_tracks_cyclist_df = gpd.read_file(cyclist_gpkg)
        result_utm_tracks_cyclist_df = gpd.read_file(result_cyclist_gpkg)

        # Raise error if df's are not equal
        assert_frame_equal(expected_utm_tracks_truck_df, result_utm_tracks_truck_df)
        assert_frame_equal(expected_utm_tracks_cyclist_df, result_utm_tracks_cyclist_df)

    def test_transform_refptsFileInSameDir_zippedOttrkFilesAsParam(
        self,
        tmp_cyclist_refpts_file: Path,
        cyclist_gpkg: Path,
        tmp_ottrk_cyclist_bz2: Path,
    ) -> None:
        # sourcery skip: remove-assert-true, remove-redundant-pass
        assert tmp_cyclist_refpts_file.exists()
        transform(paths=[tmp_ottrk_cyclist_bz2])

        result_cyclist_gpkg_bzipped = tmp_ottrk_cyclist_bz2.with_suffix(".gpkg")

        expected_utm_tracks_cyclist_df = gpd.read_file(cyclist_gpkg)
        result_utm_tracks_cyclist_df = gpd.read_file(
            result_cyclist_gpkg_bzipped, compression="bz2"
        )

        assert_frame_equal(expected_utm_tracks_cyclist_df, result_utm_tracks_cyclist_df)

    def test_transform_noExistingRefptsFileAsParam_raiseFileNotFoundError(
        self, tmp_ottrk_cyclist_bz2: Path
    ) -> None:
        with pytest.raises(
            FileNotFoundError,
            match=r"^No reference points file with filetype:.* found!",
        ):
            transform(
                [tmp_ottrk_cyclist_bz2], refpts_file=Path("no/existing/refpts.otrfpts")
            )

    def test_transform_noRefptsFileInSameDirAsParam_raiseFileNotFoundError(
        self, tmp_ottrk_cyclist_bz2: Path
    ) -> None:
        with pytest.raises(
            FileNotFoundError,
            match=r"^No reference points file found for tracks file:.*!",
        ):
            transform([tmp_ottrk_cyclist_bz2])

    def test_transform_emptyListAsParam_raiseFileNotFoundError(
        self, single_refpts_file: Path
    ) -> None:
        with pytest.raises(
            FileNotFoundError,
            match=r"No files of type .* found to transform!",
        ):
            transform([], refpts_file=single_refpts_file)
