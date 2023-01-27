from pathlib import Path

import pytest
import torch
from cv2 import VideoCapture

from OTVision.detect.yolo import (
    YOLOv5ModelNotFoundError,
    _get_batch_of_frames,
    _load_custom_model,
    _load_existing_model,
    loadmodel,
)


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


class TestLoadModel:
    CONF_THRESH: float = 0.25
    IOU_THRESH: float = 0.25

    @pytest.fixture
    def text_file(self, test_data_tmp_dir: Path) -> Path:
        text_file = Path(test_data_tmp_dir, "text_file.txt")
        text_file.touch(exist_ok=True)
        return text_file

    def test_load_existing_model_notExistingModelName_raiseAttributeException(
        self,
    ) -> None:
        with pytest.raises(YOLOv5ModelNotFoundError):
            _load_existing_model("NotExistingModelName", False)

    def test_load_existing_model_withCorrectParams(self) -> None:
        model = _load_existing_model("yolov5s", False)
        assert isinstance(model, torch.nn.Module)

    def test_load_custom_model_notAPtFileAsParam_raiseAttributeError(
        self, text_file: Path
    ) -> None:
        with pytest.raises(
            ValueError, match=f"Weights at '{text_file}' is not a pt file!"
        ):
            _load_custom_model(text_file, False)

    def test_load_model_notExistingModelName_raiseYOLOv5ModelNotFoundError(
        self,
    ) -> None:
        model_name = "NotExistingModelName"
        with pytest.raises(YOLOv5ModelNotFoundError):
            loadmodel(model_name, self.CONF_THRESH, self.IOU_THRESH)

    def test_load_model_notAPtFileAsParam_raiseAttributeError(
        self, text_file: Path
    ) -> None:
        with pytest.raises(ValueError, match=r"Weights at '.*' is not a pt file!"):
            loadmodel(str(text_file), self.CONF_THRESH, self.IOU_THRESH)
