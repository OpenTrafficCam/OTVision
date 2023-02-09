import shutil
from pathlib import Path
from typing import Generator, TypeVar

import cv2
import numpy as np
import pytest

from OTVision.config import CONFIG
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


def test_check_ffmpeg() -> None:
    """Tests if ffmpeg can be called as a subprocess"""
    check_ffmpeg()


@pytest.mark.parametrize(
    "test_case, output_filetype, input_fps, fps_from_filename",
    [
        ("default", OUTPUT_FILETYPE, INPUT_FPS, FPS_FROM_FILENAME),
        ("fps_from_filename", OUTPUT_FILETYPE, 180, True),
        ("input_fps_20", OUTPUT_FILETYPE, 20.0, False),
        ("input_fps_40", OUTPUT_FILETYPE, 40.0, False),
        ("output_filetype_avi", ".avi", INPUT_FPS, FPS_FROM_FILENAME),
        ("output_filetype_mkv", ".mkv", INPUT_FPS, FPS_FROM_FILENAME),
        ("output_filetype_mov", ".mov", INPUT_FPS, FPS_FROM_FILENAME),
        ("output_filetype_mp4", ".mp4", INPUT_FPS, FPS_FROM_FILENAME),
    ],
)
def test_pass_convert(
    test_convert_dir: Path,
    test_convert_tmp_dir: Path,
    test_case: str,
    output_filetype: str,
    input_fps: float,
    fps_from_filename: bool,
) -> None:
    """Tests the main function of OTVision/convert/convert.py
    transforming short test videos from h264 to mp4 based on
    framerate specified as part of the file path using ffmpeg.exe
    """

    # # Convert test h264 without further arguments
    convert(
        paths=[test_convert_tmp_dir / test_case],
        output_filetype=output_filetype,
        input_fps=input_fps,
        fps_from_filename=fps_from_filename,
    )

    # Local function to turn videos into np arrays for comparison
    def array_from_video(file: Path) -> np.ndarray:
        """Turn videos into np arrays for comparison

        Args:
            path (Path): Path to the video file

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

    # Get reference video files
    ref_video_files = list((test_convert_dir / test_case).glob(f"*{output_filetype}"))

    if not ref_video_files:
        raise FileNotFoundError(
            f"No reference video files found in {test_convert_dir / test_case}"
        )

    for ref_video_file in ref_video_files:
        test_video_file = test_convert_tmp_dir / test_case / ref_video_file.name

        # Assert shapes of ref and test video arrays

        array_ref_video = array_from_video(ref_video_file)
        array_test_video = array_from_video(test_video_file)

        assert array_ref_video.shape == array_test_video.shape

        # Assert size of ef and test video files

        assert ref_video_file.stat().st_size == pytest.approx(
            test_video_file.stat().st_size, rel=0.01
        )

    # BUG: Video files converted from h264 are different on each platform
    # # Compare all test video files to their respective reference video files
    # video_file_names = [file.name for file in ref_video_files]
    # equal_files, different_files, irregular_files = cmpfiles(
    #     a=test_convert_dir / test_case,
    #     b=test_convert_tmp_dir / test_case,
    #     common=video_file_names,
    #     shallow=False,
    # )
    # for equal_file in equal_files:
    #     assert equal_file in video_file_names
    # assert not different_files
    # assert not irregular_files


@pytest.mark.parametrize("delete_input", [(True), (False)])
def test_pass_convert_delete_input(
    test_convert_tmp_dir: Path, delete_input: bool
) -> None:
    """Tests if the main function of OTVision/convert/convert.py properly deletes
    input h264 files or not based on the delete_input parameter"""

    test_case = "default"
    extension = ".h264"

    # Get all h264 files to test for
    pre_test_h264_files = list((test_convert_tmp_dir / test_case).glob(f"*{extension}"))

    # Convert all test h264 files
    convert(paths=[test_convert_tmp_dir / test_case], delete_input=delete_input)

    # Get all h264 files to test for
    post_test_h264_files = list(
        (test_convert_tmp_dir / test_case).glob(f"*{extension}")
    )

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
    test_video_files = (test_convert_tmp_dir / test_case).glob(f"*{extension}")

    # Convert all test h264 files for a first time and get file statistics
    convert(paths=[test_convert_tmp_dir / test_case])
    pre_test_file_stats = [file.stat().st_mtime_ns for file in test_video_files]

    # Convert all test h264 files for a second time and get file statistics
    convert(paths=[test_convert_tmp_dir / test_case], overwrite=overwrite)
    post_test_file_stats = [file.stat().st_mtime_ns for file in test_video_files]

    # Check if file statistics are different
    for pre, post in zip(pre_test_file_stats, post_test_file_stats):
        if overwrite:
            assert pre != post
        else:
            assert pre == post


def test_fail_fps_from_filename(test_convert_tmp_dir: Path) -> None:
    """Tests if the correct ValueError is raised if the main function of
    OTVision/convert/convert.py is called with fps_from_filename == True but
    h264 file names do not contain the frame rate as specified in helpers/formats.py"""

    test_case = "fail_fps_from_filename"

    with pytest.raises(ValueError, match="Cannot read frame rate from file name*."):
        convert(paths=[test_convert_tmp_dir / test_case], fps_from_filename=True)
