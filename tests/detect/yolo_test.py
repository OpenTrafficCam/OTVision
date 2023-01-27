from pathlib import Path

import pytest
from cv2 import VideoCapture

from OTVision.detect.yolo import _get_batch_of_frames


@pytest.fixture
def video_path() -> str:
    return str(Path(__file__).parents[1] / "data" / "Testvideo_FR20_Cars-Cyclist.mp4")


@pytest.fixture
def num_frames(video_path: str) -> int:
    cap = VideoCapture(video_path)

    len_frames = 0

    while True:
        gotframe, _ = cap.read()
        if not gotframe:
            break
        len_frames += 1
    cap.release()

    return len_frames


def test_get_batch_of_frames_chunksizeGreaterThanVideoFramesParam(
    video_path: str, num_frames: int
) -> None:
    cap = VideoCapture(video_path)
    gotframe, frames = _get_batch_of_frames(cap, num_frames + 1)
    assert len(frames) == num_frames
    assert gotframe is False


def test_get_batch_of_frames_chunksizeZero(video_path: str) -> None:
    cap = VideoCapture(video_path)
    gotframe, frames = _get_batch_of_frames(cap, 0)
    assert len(frames) == 0
    assert gotframe is False


def test_get_batch_of_frames_chunksize5(video_path: str) -> None:
    chunksize = 5
    cap = VideoCapture(video_path)
    while True:
        gotframe, frames = _get_batch_of_frames(cap, chunksize)

        if not frames:
            break

        if gotframe:
            assert len(frames) == chunksize
        else:
            assert len(frames) < chunksize
            assert len(frames) >= 0


def test_get_batch_of_frames_chunksize0(video_path: str) -> None:
    chunksize = 0
    cap = VideoCapture(video_path)
    gotframe, frames = _get_batch_of_frames(cap, chunksize)
    assert gotframe is False
    assert not frames


if __name__ == "__main__":
    video_path()
