from pathlib import Path
from unittest.mock import Mock

import numpy
import pytest
from cv2 import VideoCapture
from numpy.testing import assert_array_equal
from torch import Tensor

from OTVision.detect.yolo import DISPLAYMATRIX, Yolov8, rotate
from OTVision.track.preprocess import Detection


@pytest.mark.parametrize(
    "angle, expected",
    [
        (90, [[2, 4], [1, 3]]),
        (-90, [[3, 1], [4, 2]]),
        (-180, [[4, 3], [2, 1]]),
        (180, [[4, 3], [2, 1]]),
    ],
)
def test_rotate(angle: int, expected: list[list[int]]) -> None:
    actual_array = numpy.array([[1, 2], [3, 4]], int)
    expected_array = numpy.array(expected, int)

    result = rotate(actual_array, {DISPLAYMATRIX: angle})

    assert_array_equal(result, expected_array)


def test_rotate_by_non_90_degree() -> None:
    actual_array = numpy.array([[1, 2], [3, 4]], int)

    with pytest.raises(ValueError):
        rotate(actual_array, {DISPLAYMATRIX: 20})


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
    def test_convert_x_y_coordinates(self) -> None:
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

        model = Yolov8(
            Mock(),
            mock_yolo,
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


class TestObjectDetection:
    def test_detection_start_and_end_are_considered(self) -> None:

        pass
