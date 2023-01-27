import pytest

from OTVision.helpers.formats import _get_datetime_from_filename, _get_fps_from_filename


def test_get_datetime_from_filename_validDateAsParam() -> None:
    file_name = "file_name_1993-01-01_13-00-00"
    expected = "1993-01-01_13-00-00"
    default_date = "1970-01-01_00-00-00"

    result = _get_datetime_from_filename(file_name, default_date)
    assert result == expected


@pytest.mark.parametrize(
    "file_name",
    [
        "fname-0000-00-00_00-00-00123123",
        "fname-13130000-00-00_00-00-00123123",
    ],
)
def test_get_datetime_from_filename_invalidDateFormatAsParam(file_name: str) -> None:
    default_date = "1970-01-01_00-00-00"

    result = _get_datetime_from_filename(file_name, default_date)
    assert result == default_date


@pytest.mark.parametrize(
    "file_name",
    [
        "fname-0000-00-00_00-00-00",
        "fname-0001-01-01-50-01-00",
        "fname_0001-13-01_01-01-01",
        "fname_0001-02-31_01-01-01",
        "fname_0001-02-31_70-01-01",
    ],
)
def test_get_datetime_from_filename_invalidDateAsParam(file_name: str) -> None:
    default_date = "1970-01-01_00-00-00"

    result = _get_datetime_from_filename(file_name, default_date)
    assert result == default_date


@pytest.mark.parametrize(
    "fname, expected",
    [
        ("fname_FR001_0001-01-01_01-01-01", 1),
        ("fname_FR400_-01-01_01-01-01", 400),
        ("fname_FR0_-01-01_01-01-01", 0),
        ("fname_FR0_-01-01_01-01-01", 0),
    ],
)
def test_get_fps_from_filename_validFpsAsParam(fname: str, expected: int) -> None:
    result = _get_fps_from_filename(fname)
    assert result == expected


@pytest.mark.parametrize(
    "fname",
    [
        "fname_FR_0001-01-01_01-01-01",
        "fname_FR0001-01-01_01-01-01",
        "fname_FR_0001-01-01_01-01-01",
        "fname_FR_",
    ],
)
def test_get_fps_from_filename_invalidFilenameAsParam(fname: str) -> None:
    with pytest.raises(ValueError):
        _get_fps_from_filename(fname)
