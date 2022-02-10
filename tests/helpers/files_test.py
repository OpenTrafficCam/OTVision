from pathlib import Path
import shutil

import pytest

from OTVision.helpers.files import is_in_format
from OTVision.helpers.files import get_files


@pytest.fixture
def test_dir_with_files():
    test_dir = Path(__file__).parents[1] / "resources" / "test_dir"
    file_names = [
        "readme.txt",
        "cities.json",
        "config.xml",
        "img_1.PNG",
        "img_2.png",
        "img_3.PnG",
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


def test_get_files_dirPathAsPosixPathObjectAs1stParam_returnsCorrectList(
    test_dir_with_files,
):
    json_file_path = str(Path(test_dir_with_files, "cities.json"))
    xml_file_path = str(Path(test_dir_with_files, "config.xml"))

    files = get_files(test_dir_with_files, [".json", ".xml"])

    assert json_file_path in files
    assert xml_file_path in files


def test_get_files_dirPathAsStringAs1stParam_returnsCorrectList(test_dir_with_files):
    json_file_path = str(Path(test_dir_with_files, "cities.json"))
    xml_file_path = str(Path(test_dir_with_files, "config.xml"))

    files = get_files(str(test_dir_with_files), [".json", ".xml"])

    assert json_file_path in files
    assert xml_file_path in files


def test_get_files_noFilenamesAs2ndParam_ReturnEmptyList(test_dir_with_files):
    files = get_files(test_dir_with_files, [])
    assert [] == files


def test_get_files_invalidTypeAs1stParam_RaiseTypeError():
    dictionary = dict()
    with pytest.raises(TypeError):
        get_files(dictionary, ".json")


def test_get_files_sameFiletypeWithDifferentCasesAsParam_returnsCorrectList(
    test_dir_with_files,
):
    img_1_path = str(Path(test_dir_with_files, "img_1.PNG"))
    img_2_path = str(Path(test_dir_with_files, "img_2.png"))
    img_3_path = str(Path(test_dir_with_files, "img_3.PnG"))

    files = get_files(test_dir_with_files, ".png")
    assert img_1_path in files
    assert img_2_path in files
    assert img_3_path in files


def test_is_in_format_fileFormatStartsWithDotAs2ndParam_returnsTrue():
    file_path = "path/to/file/f.yaml"
    file_formats = [".yaml"]
    assert is_in_format(file_path, file_formats)


def test_is_in_format_fileFormatStartsWithoutDotAs2ndParam_returnsTrue():
    file_path = "path/to/file/f.yaml"
    file_formats = ["yaml"]
    assert is_in_format(file_path, file_formats)


def test_is_in_format_fileWithNotDefinedFileFormatAs1stParam_returnsFalse():
    file_path = "path/to/file/f.yaml"
    file_formats = [".txt"]
    assert not is_in_format(file_path, file_formats)


def test_is_in_format_filesWithFileFormatsWithDifferentCasesAs1stParam_returns_True():
    yaml_1 = "path/to/file/f.yaml"
    yaml_2 = "path/to/file/f.YAML"
    yaml_3 = "path/to/file/f.yAmL"
    file_formats = [".yaml"]
    assert is_in_format(yaml_1, file_formats)
    assert is_in_format(yaml_2, file_formats)
    assert is_in_format(yaml_3, file_formats)
