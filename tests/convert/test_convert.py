import os
import shutil
from pathlib import Path

from OTVision.config import CONFIG
from OTVision.convert.convert import check_ffmpeg
from OTVision.convert.convert import main as convert
from OTVision.helpers.files import get_files
from OTVision.helpers.machine import ON_WINDOWS


def test_check_ffmpeg():
    """Tests if ffmpeg.exe is checked and downloaded correctly"""

    if not ON_WINDOWS:
        return

    ffmpeg_exe_path = Path(CONFIG["CONVERT"]["FFMPEG_PATH"])

    # Deletes ffmpeg.exe, tests if missing ffmpeg.exe is detected and downloaded again
    if ffmpeg_exe_path.is_file():
        ffmpeg_creation_time_before = os.path.getctime(ffmpeg_exe_path)
    else:
        ffmpeg_creation_time_before = 0
    ffmpeg_exe_path.unlink(missing_ok=True)  # Delete ffmpeg.exe
    check_ffmpeg()
    ffmpeg_creation_time_after = os.path.getctime(ffmpeg_exe_path)
    assert ffmpeg_creation_time_before != ffmpeg_creation_time_after
    assert ffmpeg_exe_path.is_file()

    # Test if exsting ffmpeg.exe is detected and therefore not downloaded again
    ffmpeg_creation_time_before = os.path.getctime(ffmpeg_exe_path)
    check_ffmpeg()
    ffmpeg_creation_time_after = os.path.getctime(ffmpeg_exe_path)
    assert ffmpeg_creation_time_before == ffmpeg_creation_time_after


def test_convert(test_data_tmp_dir: Path, test_data_dir: Path):
    """Tests the main function of OTVision/convert/convert.py
    transforming short test videos from h264 to mp4 based on
    framerate specified as part of the file path using ffmpeg.exe
    """

    if not ON_WINDOWS:
        return

    # Get reference data
    h264_ref_videos = get_files(paths=[test_data_dir], filetypes=[".h264"])
    mp4_ref_videos = get_files(paths=[test_data_dir], filetypes=[".mp4"])

    # Copy input data to temporary folder and start conversion
    for h264_ref_video in h264_ref_videos:
        shutil.copy2(
            Path(h264_ref_video),
            Path(h264_ref_video).parents[1]
            / test_data_tmp_dir.name
            / Path(h264_ref_video).name,
        )
    h264_test_videos = get_files(
        paths=[test_data_tmp_dir],
        filetypes=[".h264"],
    )
    convert(h264_test_videos)

    # Local function to turn videos into np arrays for comparison
    import cv2
    import numpy as np

    def array_from_video(path: Path):
        frames = []
        cap = cv2.VideoCapture(str(path))
        ret = True
        while ret:
            (
                ret,
                img,
            ) = cap.read()
            if ret:
                frames.append(img)
        return np.stack(frames, axis=0)

    # Turn test and reference mp4 videos into np arrays and compare them
    # IDEA: Just test if file sizes of both mp4 files are equal
    for mp4_ref_video in mp4_ref_videos:
        assert np.array_equal(
            array_from_video(mp4_ref_video),
            array_from_video(
                test_data_tmp_dir
                / mp4_ref_video.name
            ),
        )
