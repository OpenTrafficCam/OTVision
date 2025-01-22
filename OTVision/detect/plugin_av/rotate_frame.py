from av import VideoFrame
from numpy import ndarray, rot90

DISPLAYMATRIX = "DISPLAYMATRIX"


class AvVideoFrameRotator:
    def __init__(self, img_format: str = "rgb24"):
        self._img_format = img_format

    def rotate(self, frame: VideoFrame, side_data: dict) -> ndarray:
        array = frame.to_ndarray(format=self._img_format)
        rotated_image = rotate(array, side_data)
        return rotated_image


def rotate(array: ndarray, side_data: dict) -> ndarray:
    """
    Rotate a numpy array using the DISPLAYMATRIX rotation angle defined in side_data.

    Args:
        array: to rotate
        side_data: metadata dictionary to read the angle from

    Returns: rotated array

    """
    if DISPLAYMATRIX in side_data:
        angle = side_data[DISPLAYMATRIX]
        if angle % 90 != 0:
            raise ValueError(
                f"Rotation angle must be multiple of 90 degrees, but is {angle}"
            )
        rotation = angle / 90
        rotated_image = rot90(array, rotation)
        return rotated_image
    return array
