from unittest.mock import Mock

import pytest
from numpy import array
from numpy.testing import assert_array_equal

from OTVision.detect.plugin_av.rotate_frame import AvVideoFrameRotator, rotate


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
    actual_array = array([[1, 2], [3, 4]], int)
    expected_array = array(expected, int)

    result = rotate(actual_array, angle)

    assert_array_equal(result, expected_array)


def test_rotate_by_non_90_degree() -> None:
    actual_array = array([[1, 2], [3, 4]], int)

    with pytest.raises(ValueError):
        rotate(actual_array, 20)


def test_rotate_frame_by_its_rotation() -> None:
    frame = Mock()
    frame.to_ndarray.return_value = array([[1, 2], [3, 4]], int)
    frame.rotation = 90

    result = AvVideoFrameRotator(img_format="rgb24").rotate(frame)

    assert_array_equal(result, array([[2, 4], [1, 3]], int))
    frame.to_ndarray.assert_called_once_with(format="rgb24")
