import os
import shutil
from pathlib import Path

import pytest

from OTVision.helpers.files import (
    get_files,
    has_filetype,
    read_json,
    replace_filetype,
    write_json,
)
from tests.conftest import YieldFixture


@pytest.fixture
def test_dir_with_files() -> YieldFixture[Path]:
    test_dir = Path(__file__).parents[1] / "resources" / "test_dir"
    file_names = [
        Path("readme.txt"),
        Path("cities.json"),
        Path("config.xml"),
        Path("img_1.PNG"),
        Path("img_2.png"),
        Path("img_3.PnG"),
    ]

    if test_dir.exists():
        shutil.rmtree(test_dir)

    # Create test directory
    test_dir.mkdir(parents=True)
    files = [Path(test_dir, name) for name in file_names]

    # Create test files
    for f in files:
        f.touch(exist_ok=True)

    yield test_dir

    # Delete directory
    shutil.rmtree(test_dir)


def test_write_compressed_json(
    test_dir_with_files: Path,
) -> None:
    nested = {f"data_{i}": str(i) for i in range(20)}
    data = {
        "some": "data",
        "other": "data",
        "nested": nested,
    }

    json_path = Path(test_dir_with_files, "json_dump.json")
    compressed_path = Path(test_dir_with_files, "compressed_dump.json.bz2")

    write_json(data, json_path, ".json", compress=False)
    write_json(data, compressed_path, ".bz2", compress=True)
    json_size = os.path.getsize(json_path)
    compressed_size = os.path.getsize(compressed_path)
    assert json_path.exists()
    assert compressed_path.exists()
    assert json_size > compressed_size


def test_read_compressed_json(
    test_dir_with_files: Path,
) -> None:
    data = {
        "some": "data",
        "other": "data",
        "nested": {"data_one": "one", "data_two": "two"},
    }

    compressed_path = Path(test_dir_with_files, "compressed_dump.json.bz2")
    write_json(data, compressed_path, ".bz2", compress=True)

    read_data = read_json(compressed_path, ".bz2", decompress=True)

    assert read_data == data


def test_get_files_dirPathAsListOfPathObjectAs1stParam_returnsCorrectList(
    test_dir_with_files: Path,
) -> None:
    json_file_path = Path(test_dir_with_files, "cities.json")
    xml_file_path = Path(test_dir_with_files, "config.xml")

    files = get_files(paths=[test_dir_with_files], filetypes=[".json", ".xml"])

    assert json_file_path in files
    assert xml_file_path in files


def test_get_files_dirPathsAsListOfString_RaiseTypeError(
    test_dir_with_files: Path,
) -> None:
    list_of_str = [
        str(Path(test_dir_with_files, "img_1.PNG")),
        Path(test_dir_with_files, "img_2.png"),
        Path(test_dir_with_files, "img_3.PnG"),
    ]

    with pytest.raises(TypeError):
        get_files(paths=list_of_str)  # type: ignore


def test_get_files_dirPathsAsParam_RaiseTypeError(test_dir_with_files: Path) -> None:
    with pytest.raises(TypeError):
        get_files(
            paths=test_dir_with_files,  # type: ignore
            filetypes=[".json", ".xml"],
        )


def test_get_files_dirPathsAsString_RaiseTypeError(test_dir_with_files: Path) -> None:
    with pytest.raises(TypeError):
        get_files(
            paths=str(test_dir_with_files),  # type: ignore
            filetypes=[".json", ".xml"],
        )


def test_get_files_noFilenamesAs2ndParam_ReturnEmptyList(
    test_dir_with_files: Path,
) -> None:
    files = get_files(paths=[test_dir_with_files], filetypes=[])
    assert [] == files


def test_get_files_invalidTypeListOfNumbersAs1stParam_RaiseTypeError() -> None:
    numbers = [1, 2, 3]
    with pytest.raises(TypeError):
        get_files(paths=numbers, filetypes=[".json"])  # type: ignore


def test_get_files_dictAs1stParam_RaiseTypeError() -> None:
    _dict: dict = {}
    with pytest.raises(TypeError):
        get_files(paths=_dict, filetypes=[".json"])  # type: ignore


def test_get_files_sameFiletypeWithDifferentCasesAsParam_returnsCorrectList(
    test_dir_with_files: Path,
) -> None:
    img_1_path = Path(test_dir_with_files, "img_1.PNG")
    img_2_path = Path(test_dir_with_files, "img_2.png")
    img_3_path = Path(test_dir_with_files, "img_3.PnG")

    files = get_files(paths=[test_dir_with_files], filetypes=[".png"])
    assert img_1_path in files
    assert img_2_path in files
    assert img_3_path in files


def test_replace_filetype_certainOldFiletype(test_dir_with_files: Path) -> None:
    files = get_files(paths=[test_dir_with_files])
    new_filetype = ".fancypng"
    replaced_files = replace_filetype(
        files=files, old_filetype=".png", new_filetype=new_filetype
    )
    for path in replaced_files:
        if path.is_file():
            assert path.suffix.lower() != ".png"


def test_replace_filetype_noOldFiletype(test_dir_with_files: Path) -> None:
    files = get_files([test_dir_with_files])
    new_filetype = ".fancypng"
    replaced_files = replace_filetype(files=files, new_filetype=new_filetype)
    for path in replaced_files:
        if path.is_file():
            assert path.suffix.lower() == new_filetype


def test_has_filetype_fileFormatStartsWithDotAs2ndParam_returnsTrue() -> None:
    file = Path("path/to/file/f.yaml")
    filetypes = [".yaml"]
    assert has_filetype(file=file, filetypes=filetypes)


def test_has_filetype_fileFormatStartsWithoutDotAs2ndParam_returnsTrue() -> None:
    file = Path("path/to/file/f.yaml")
    filetypes = ["yaml"]
    assert has_filetype(file=file, filetypes=filetypes)


def test_has_filetype_fileWithNotDefinedFileFormatAs1stParam_returnsFalse() -> None:
    file = Path("path/to/file/f.yaml")
    filetypes = [".txt"]
    assert not has_filetype(file=file, filetypes=filetypes)


def test_has_filetype_fileFormatsWithDifferentCaseAsParam_returns_True() -> None:
    yaml_1 = Path("path/to/file/f.yaml")
    yaml_2 = Path("path/to/file/f.YAML")
    yaml_3 = Path("path/to/file/f.yAmL")
    filetypes = [".yaml"]
    assert has_filetype(file=yaml_1, filetypes=filetypes)
    assert has_filetype(file=yaml_2, filetypes=filetypes)
    assert has_filetype(file=yaml_3, filetypes=filetypes)
