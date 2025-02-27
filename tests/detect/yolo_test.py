from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from cv2 import VideoCapture
from torch import Tensor

from OTVision.detect.yolo import Yolov8
from OTVision.track.model.detection import Detection


@pytest.fixture
def video_path() -> str:
    return str(
        Path(__file__).parents[1]
        / "data"
        / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"
    )


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


class TestConvertDetections:
    @patch.object(Yolov8, "_load_model")
    def test_convert_x_y_coordinates(self, mock_load_model: Mock) -> None:
        classification: int = 1
        name: str = "name"
        names = {classification: name}
        x_input = 20
        x_output = 15
        y_input = 20
        y_output = 15
        width = 10
        height = 10
        confidence = 0.5

        mock_yolo = Mock().return_value
        mock_yolo.names = names
        mock_load_model.return_value = mock_yolo

        model = Yolov8(
            Mock(),
            confidence=0.25,
            iou=0.25,
            img_size=640,
            half_precision=False,
            normalized=False,
        )

        result = model._parse_detection(
            Tensor([x_input, y_input, width, height]), classification, confidence
        )

        assert result == Detection(name, confidence, x_output, y_output, width, height)
