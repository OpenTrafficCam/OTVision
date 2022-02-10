from pathlib import Path
import shutil

import pytest

from OTVision.helpers.files import is_in_format
from OTVision.helpers.files import get_files


@pytest.fixture
def test_dir_with_files():
    test_dir = Path(__file__).parents[1] / "resources" / "test_dir"
    file_names = ["readme.txt", "cities.json", "config.xml", "img_1.PNG", "img_1.png"]

    # Create test directory
    test_dir.mkdir(parents=True)
    files = [Path(test_dir, name) for name in file_names]

    # Create test files
    for f in files:
        f.touch(exist_ok=True)

    yield test_dir

    # Delete directory
    shutil.rmtree(test_dir)


def test_get_files_correctParam_returnsCorrectList(test_dir_with_files):
    json_file_path = str(Path(test_dir_with_files, "cities.json"))
    xml_file_path = str(Path(test_dir_with_files, "config.xml"))

    files = get_files(test_dir_with_files, [".json", ".xml"])

    assert json_file_path in files
    assert xml_file_path in files


def test_get_files_noFilenamesAs2ndParam_ReturnEmptyList(test_dir_with_files):
    files = get_files(test_dir_with_files, [])
    assert [] == files


def test_get_files_invalidTypeAs1stParam_RaiseTypeError():
    dictionary = dict()
    with pytest.raises(TypeError):
        get_files(dictionary, ".json")
