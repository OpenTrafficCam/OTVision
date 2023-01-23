import shutil
from pathlib import Path

import cv2
import numpy as np
from pytest import approx

from OTVision.convert.convert import check_ffmpeg
from OTVision.convert.convert import main as convert
from OTVision.helpers.files import get_files


def test_check_ffmpeg() -> None:
    """Tests if ffmpeg can be called as a subprocess"""
    check_ffmpeg()


def test_convert(test_data_tmp_dir: Path, test_data_dir: Path) -> None:
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

    for mp4_ref_video in mp4_ref_videos:

        mp4_test_video = test_data_tmp_dir / mp4_ref_video.name

        # Assert shapes of ref and test video arrays

        array_mp4_ref_video = array_from_video(mp4_ref_video)
        array_mp4_test_video = array_from_video(mp4_test_video)

        assert array_mp4_ref_video.shape == array_mp4_test_video.shape

        # Assert size of ef and test video files

        assert mp4_ref_video.stat().st_size == approx(
            mp4_test_video.stat().st_size, rel=0.01
        )

        # TODO: Assert ref and test video array´s color values
        # (for exact assertion use assert np.array_equal())
        # import numpy.testing as np_tst
        # precision = 5
        # np_tst.assert_array_almost_equal(
        #     array_mp4_ref_video,
        #     array_mp4_test_video,
        #     decimal=precision,
        # )
