import os
import shutil
from pathlib import Path
from typing import Generator, TypeVar

import cv2
import numpy as np
import pytest

from OTVision.config import CONFIG, FILETYPES, VID_ROTATABLE
from OTVision.convert.convert import check_ffmpeg
from OTVision.convert.convert import main as convert

OUTPUT_FILETYPE = CONFIG["CONVERT"]["OUTPUT_FILETYPE"]
INPUT_FPS = CONFIG["CONVERT"]["INPUT_FPS"]
OUTPUT_FPS = CONFIG["CONVERT"]["OUTPUT_FPS"]
FPS_FROM_FILENAME = CONFIG["CONVERT"]["FPS_FROM_FILENAME"]
DELETE_INPUT = CONFIG["CONVERT"]["DELETE_INPUT"]
OVERWRITE = CONFIG["CONVERT"]["OVERWRITE"]

T = TypeVar("T")
YieldFixture = Generator[T, None, None]


@pytest.fixture
def test_convert_dir(test_data_dir: Path) -> Path:
    return test_data_dir / "convert"


@pytest.fixture
def test_convert_tmp_dir(
    test_data_tmp_dir: Path, test_convert_dir: Path
) -> YieldFixture[Path]:
    # Create tmp dir
    test_convert_tmp_dir = test_data_tmp_dir / "convert"
    test_convert_tmp_dir.mkdir(exist_ok=True)
    # Copy test files to tmp dir
    shutil.copytree(test_convert_dir, test_convert_tmp_dir, dirs_exist_ok=True)
    # Delete video files from tmp dir (we create them during the tests)
    extension = CONFIG["DEFAULT_FILETYPE"]["VID"]
    video_files_to_delete = test_convert_tmp_dir.rglob(f"*{extension}")
    for f in video_files_to_delete:
        f.unlink()
    # Yield tmp dir
    yield test_convert_tmp_dir
    # Teardown tmp dir after use in test
    shutil.rmtree(test_convert_tmp_dir)


@pytest.fixture
def convert_rotate_tmp_dir(
    test_data_tmp_dir: Path, test_convert_dir: Path
) -> YieldFixture[Path]:
    test_convert_tmp_dir = test_data_tmp_dir / "convert" / "rotation"
    test_convert_tmp_dir.mkdir(exist_ok=True, parents=True)
    h264files = (test_convert_dir / "rotation").glob("*.h264")
    for h264file in h264files:
        shutil.copy2(h264file, test_convert_tmp_dir / h264file.name)
    yield test_convert_tmp_dir
    shutil.rmtree(test_convert_tmp_dir)


@pytest.fixture
def reference_mp4_files(test_convert_dir: Path) -> list[Path]:
    cyclist_mp4 = (
        test_convert_dir
        / "output_filetype_mp4/Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"
    )
    truck_mp4 = (
        test_convert_dir
        / "output_filetype_mp4/Testvideo_Cars-Truck_FR20_2020-01-01_00-00-00.mp4"
    )
    return [cyclist_mp4, truck_mp4]


def test_check_ffmpeg() -> None:
    """Tests if ffmpeg can be called as a subprocess"""
    check_ffmpeg()


@pytest.mark.parametrize(
    "test_case,  input_fps, fps_from_filename",
    [
        ("default", INPUT_FPS, FPS_FROM_FILENAME),
        ("fps_from_filename", 180, True),
        ("input_fps_20", 20.0, False),
        ("input_fps_40", 40.0, False),
        ("output_filetype_avi", INPUT_FPS, FPS_FROM_FILENAME),
        ("output_filetype_mkv", INPUT_FPS, FPS_FROM_FILENAME),
        ("output_filetype_mov", INPUT_FPS, FPS_FROM_FILENAME),
        ("output_filetype_mp4", INPUT_FPS, FPS_FROM_FILENAME),
    ],
)
def test_pass_convert(
    test_convert_tmp_dir: Path,
    test_case: str,
    input_fps: float,
    fps_from_filename: bool,
    reference_mp4_files: list[Path],
) -> None:
    """Tests the main function of OTVision/convert/convert.py
    transforming short test videos from h264 to mp4 based on
    framerate specified as part of the file path using ffmpeg.exe
    """

    # Build test dir paths
    test_case_tmp_dir = test_convert_tmp_dir / test_case

    # Convert test h264 files
    convert(
        paths=[test_case_tmp_dir],
        output_filetype=OUTPUT_FILETYPE,
        input_fps=input_fps,
        fps_from_filename=fps_from_filename,
    )

    # Get reference video files
    for ref_video_file in reference_mp4_files:
        test_video_file = test_case_tmp_dir / ref_video_file.name
        assert_videos_are_equal(actual=test_video_file, expected=ref_video_file)

    # BUG: Video files converted from h264 are different on each platform
    # # Compare all test video files to their respective reference video files
    # video_file_names = [file.name for file in ref_video_files]
    # equal_files, different_files, irregular_files = cmpfiles(
    #     a=test_case_dir,
    #     b=test_case_tmp_dir,
    #     common=video_file_names,
    #     shallow=False,
    # )
    # for equal_file in equal_files:
    #     assert equal_file in video_file_names
    # assert not different_files
    # assert not irregular_files


def assert_videos_are_equal(actual: Path, expected: Path) -> None:
    # Assert shapes of ref and test video arrays

    actual_array = array_from_video(actual)
    expected_array = array_from_video(expected)

    assert actual_array.shape == expected_array.shape
    assert (actual_array == expected_array).all()

    # Assert size of ef and test video files

    assert actual.stat().st_size == pytest.approx(expected.stat().st_size, rel=0.001)

    # Local function to turn videos into np arrays for comparison


def array_from_video(file: Path) -> np.ndarray:
    """Turn videos into np arrays for comparison

    Args:
        file (Path): Path to the video file

    Returns:
        np.ndarray: Video file as numpy array
    """
    frames = []
    cap = cv2.VideoCapture(str(file))
    ret = True
    while ret:
        (
            ret,
            img,
        ) = cap.read()
        if ret:
            frames.append(img)
    return np.stack(frames, axis=0)


@pytest.mark.parametrize(
    "video_folder, output_filetype, rotation",
    [
        ("rotation", ".avi", 90),
        ("rotation", ".mkv", 90),
        ("rotation", ".mov", 90),
        ("rotation", ".mp4", 90),
    ],
)
def test_convert_rotate(
    test_convert_dir: Path,
    convert_rotate_tmp_dir: Path,
    video_folder: str,
    output_filetype: str,
    rotation: int,
) -> None:
    test_case_dir = test_convert_dir / video_folder
    test_case_tmp_dir = convert_rotate_tmp_dir

    if output_filetype in CONFIG[FILETYPES][VID_ROTATABLE]:
        convert(
            paths=[test_case_tmp_dir],
            output_filetype=output_filetype,
            rotation=rotation,
        )

        ref_video_files = list(test_case_dir.glob(f"*{output_filetype}"))

        if not ref_video_files:
            raise FileNotFoundError(
                f"No reference video files found in {test_case_dir}"
            )

        for ref_video_file in ref_video_files:
            test_video_file = test_case_tmp_dir / ref_video_file.name
            assert_videos_are_equal(actual=test_video_file, expected=ref_video_file)
    else:
        with pytest.raises(TypeError):
            convert(
                paths=[test_case_tmp_dir],
                output_filetype=output_filetype,
                rotation=rotation,
            )


@pytest.mark.parametrize("delete_input", [(True), (False)])
def test_pass_convert_delete_input(
    test_convert_tmp_dir: Path, delete_input: bool
) -> None:
    """Tests if the main function of OTVision/convert/convert.py properly deletes
    input h264 files or not based on the delete_input parameter"""

    test_case = "default"
    extension = ".h264"
    test_case_tmp_dir = test_convert_tmp_dir / test_case

    # Get all h264 files to test for
    pre_test_h264_files = list(test_case_tmp_dir.glob(f"*{extension}"))

    # Convert all test h264 files
    convert(paths=[test_case_tmp_dir], delete_input=delete_input)

    # Get all h264 files to test for
    post_test_h264_files = list(test_case_tmp_dir.glob(f"*{extension}"))

    # Check if 264 still exists
    if delete_input:
        assert not post_test_h264_files
    else:
        for pre, post in zip(pre_test_h264_files, post_test_h264_files):
            assert pre == post


@pytest.mark.parametrize("overwrite", [(True), (False)])
def test_pass_convert_overwrite(test_convert_tmp_dir: Path, overwrite: bool) -> None:
    """Tests if the main function of OTVision/convert/convert.py properly overwrites
    existing files or not based on the overwrite parameter"""

    # Get video files to test for
    test_case = "default"
    extension = CONFIG["DEFAULT_FILETYPE"]["VID"]
    test_case_tmp_dir = test_convert_tmp_dir / test_case
    test_video_files = test_case_tmp_dir.glob(f"*{extension}")

    # Convert all test h264 files for a first time and get file statistics
    convert(paths=[test_case_tmp_dir])
    pre_test_file_stats = [file.stat().st_mtime_ns for file in test_video_files]

    # Convert all test h264 files for a second time and get file statistics
    convert(paths=[test_case_tmp_dir], overwrite=overwrite)
    post_test_file_stats = [file.stat().st_mtime_ns for file in test_video_files]

    # Check if file statistics are different
    for pre, post in zip(pre_test_file_stats, post_test_file_stats):
        if overwrite:
            assert pre != post
        else:
            assert pre == post


def test_fail_convert_fps_from_filename(test_convert_tmp_dir: Path) -> None:
    """Tests if the correct ValueError is raised if the main function of
    OTVision/convert/convert.py is called with fps_from_filename == True but
    h264 file names do not contain the frame rate as specified in helpers/formats.py"""

    test_case = "fail_fps_from_filename"

    with pytest.raises(ValueError, match="Cannot read frame rate from file name*."):
        convert(paths=[test_convert_tmp_dir / test_case], fps_from_filename=True)


@pytest.mark.parametrize(
    "paths",
    [
        (1),
        ("some_str"),
        (Path("some_str")),
        ([Path("some_str"), Path("some_other_str")]),
    ],
)
def test_fail_convert_wrong_paths(paths) -> None:  # type: ignore
    """Tests if the main function of OTVision/convert/convert.py raises specific errors
    when wrong paths are given"""

    # Check if TypeError is raised
    with pytest.raises(TypeError, match=r"Paths needs to be a list of pathlib.Path"):
        convert(paths=paths)


@pytest.mark.parametrize(
    "output_filetype",
    [".foo", ".jpg", ".mpeg", ".avchd", ".flv", ".swf", ".m4v", ".mpg", ".wmv"],
)
def test_fail_convert_not_supported_output_filetypes(
    test_convert_tmp_dir: Path,
    output_filetype: str,
) -> None:
    """Tests if the main function of OTVision/convert/convert.py raises specific
    errors when not defined output filetypes are given"""

    test_case = "default"

    # Track all test detections files
    with pytest.raises(TypeError, match="Output video filetype.*"):
        convert(
            paths=[test_convert_tmp_dir / test_case],
            output_filetype=output_filetype,
        )


@pytest.mark.parametrize(
    "output_filetype, input_fps, fps_from_filename, overwrite, delete_input",
    [
        (22, INPUT_FPS, FPS_FROM_FILENAME, OVERWRITE, DELETE_INPUT),
        (OUTPUT_FILETYPE, "foo", FPS_FROM_FILENAME, OVERWRITE, DELETE_INPUT),
        (OUTPUT_FILETYPE, [40], FPS_FROM_FILENAME, OVERWRITE, DELETE_INPUT),
        (OUTPUT_FILETYPE, INPUT_FPS, 20, OVERWRITE, DELETE_INPUT),
        (OUTPUT_FILETYPE, INPUT_FPS, "foo", OVERWRITE, DELETE_INPUT),
        (OUTPUT_FILETYPE, INPUT_FPS, FPS_FROM_FILENAME, 20, DELETE_INPUT),
        (OUTPUT_FILETYPE, INPUT_FPS, FPS_FROM_FILENAME, "foo", DELETE_INPUT),
        (OUTPUT_FILETYPE, INPUT_FPS, FPS_FROM_FILENAME, OVERWRITE, 20),
        (OUTPUT_FILETYPE, INPUT_FPS, FPS_FROM_FILENAME, OVERWRITE, "foo"),
    ],
)
def test_fail_convert_wrong_parameters(
    test_convert_tmp_dir: Path,
    output_filetype: str,
    input_fps: float,
    fps_from_filename: bool,
    overwrite: bool,
    delete_input: bool,
) -> None:
    """Tests if the main function of OTVision/convert/convert.py raises specific
    errors when wrong parameters are given"""

    test_case = "default"

    # Track all test detections files
    with pytest.raises(ValueError, match=".*has to be.*"):
        convert(
            paths=[test_convert_tmp_dir / test_case],
            output_filetype=output_filetype,
            input_fps=input_fps,
            fps_from_filename=fps_from_filename,
            overwrite=overwrite,
            delete_input=delete_input,
        )


def test_convert_emptyDirAsParam(test_convert_tmp_dir: Path) -> None:
    empty_dir = test_convert_tmp_dir / "empty"
    empty_dir.mkdir()

    convert(paths=[empty_dir])

    assert os.listdir(empty_dir) == []
