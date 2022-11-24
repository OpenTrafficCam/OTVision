import shutil
from pathlib import Path

import pytest

from OTVision.helpers.files import get_files, has_filetype, replace_filetype


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


def test_get_files_dirPathAsListOfPosixPathObjectAs1stParam_returnsCorrectList(
    test_dir_with_files,
):
    json_file_path = Path(test_dir_with_files, "cities.json")
    xml_file_path = Path(test_dir_with_files, "config.xml")

    files = get_files(paths=[test_dir_with_files], filetypes=[".json", ".xml"])

    assert json_file_path in files
    assert xml_file_path in files


def test_get_files_dirPathsAsListOfString_RaiseTypeError(test_dir_with_files):
    list_of_str = [
        str(Path(test_dir_with_files, "img_1.PNG")),
        Path(test_dir_with_files, "img_2.png"),
        Path(test_dir_with_files, "img_3.PnG"),
    ]

    with pytest.raises(TypeError):
        get_files(paths=list_of_str)


def test_get_files_dirPathsAsPosixPath_RaiseTypeError(test_dir_with_files):
    with pytest.raises(TypeError):
        get_files(paths=test_dir_with_files, filetypes=[".json", ".xml"])


def test_get_files_dirPathsAsString_RaiseTypeError(test_dir_with_files):
    with pytest.raises(TypeError):
        get_files(paths=str(test_dir_with_files), filetypes=[".json", ".xml"])


def test_get_files_noFilenamesAs2ndParam_ReturnEmptyList(test_dir_with_files):
    files = get_files(paths=[test_dir_with_files], filetypes=[])
    assert [] == files


def test_get_files_invalidTypeListOfNumbersAs1stParam_RaiseTypeError():
    numbers = [1, 2, 3]
    with pytest.raises(TypeError):
        get_files(paths=numbers, filetypes=[".json"])


def test_get_files_dictAs1stParam_RaiseTypeError():
    _dict = {}
    with pytest.raises(TypeError):
        get_files(paths=_dict, filetypes=[".json"])


def test_get_files_sameFiletypeWithDifferentCasesAsParam_returnsCorrectList(
    test_dir_with_files,
):
    img_1_path = Path(test_dir_with_files, "img_1.PNG")
    img_2_path = Path(test_dir_with_files, "img_2.png")
    img_3_path = Path(test_dir_with_files, "img_3.PnG")

    files = get_files(paths=[test_dir_with_files], filetypes=[".png"])
    assert img_1_path in files
    assert img_2_path in files
    assert img_3_path in files


def test_replace_filetype_certainOldFiletype(test_dir_with_files):
    files = get_files(paths=[test_dir_with_files])
    new_filetype = ".fancypng"
    replaced_files = replace_filetype(
        files=files, old_filetype=".png", new_filetype=new_filetype
    )
    for path in replaced_files:
        if path.is_file():
            assert path.suffix.lower() != ".png"


def test_replace_filetype_noOldFiletype(test_dir_with_files):
    files = get_files([test_dir_with_files])
    new_filetype = ".fancypng"
    replaced_files = replace_filetype(files=files, new_filetype=new_filetype)
    for path in replaced_files:
        if path.is_file():
            assert path.suffix.lower() == new_filetype


def test_has_filetype_fileFormatStartsWithDotAs2ndParam_returnsTrue():
    file = Path("path/to/file/f.yaml")
    filetypes = [".yaml"]
    assert has_filetype(file=file, filetypes=filetypes)


def test_has_filetype_fileFormatStartsWithoutDotAs2ndParam_returnsTrue():
    file = Path("path/to/file/f.yaml")
    filetypes = ["yaml"]
    assert has_filetype(file=file, filetypes=filetypes)


def test_has_filetype_fileWithNotDefinedFileFormatAs1stParam_returnsFalse():
    file = Path("path/to/file/f.yaml")
    filetypes = [".txt"]
    assert not has_filetype(file=file, filetypes=filetypes)


def test_has_filetype_filesWithFileFormatsWithDifferentCasesAs1stParam_returns_True():
    yaml_1 = Path("path/to/file/f.yaml")
    yaml_2 = Path("path/to/file/f.YAML")
    yaml_3 = Path("path/to/file/f.yAmL")
    filetypes = [".yaml"]
    assert has_filetype(file=yaml_1, filetypes=filetypes)
    assert has_filetype(file=yaml_2, filetypes=filetypes)
    assert has_filetype(file=yaml_3, filetypes=filetypes)
