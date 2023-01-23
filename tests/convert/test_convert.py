import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import numpy.testing as np_tst

from OTVision.config import CONFIG
from OTVision.convert.convert import check_ffmpeg
from OTVision.convert.convert import main as convert
from OTVision.helpers.files import get_files


def test_check_ffmpeg():
    """Tests if ffmpeg can be called as a subprocess"""
    check_ffmpeg()


def test_convert(test_data_tmp_dir: Path, test_data_dir: Path):
    """Tests the main function of OTVision/convert/convert.py
    transforming short test videos from h264 to mp4 based on
    framerate specified as part of the file path using ffmpeg.exe
    """

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
    def array_from_video(path: Path) -> np.ndarray:
        """Turn videos into np arrays for comparison

        Args:
            path (Path): Path to the video file

        Returns:
            np.ndarray: Video file as numpy array
        """
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
    # Other idea: Just test if file sizes of both mp4 files are equal
    precision = 5
    for mp4_ref_video in mp4_ref_videos:

        mp4_test_video = test_data_tmp_dir / mp4_ref_video.name

        # Assert ref and test video (for exact assertion use assert np.array_equal())
        np_tst.assert_array_almost_equal(
            array_from_video(mp4_ref_video),
            array_from_video(mp4_test_video),
            decimal=precision,
        )
