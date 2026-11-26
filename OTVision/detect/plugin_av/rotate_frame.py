from av import VideoFrame
from numpy import ndarray, rot90


class AvVideoFrameRotator:
    def __init__(self, img_format: str = "rgb24"):
        self._img_format = img_format

    def rotate(self, frame: VideoFrame) -> ndarray:
        array = frame.to_ndarray(format=self._img_format)
        rotated_image = rotate(array, frame.rotation)
        return rotated_image


def rotate(array: ndarray, angle: int) -> ndarray:
    """
    Rotate a numpy array by the display rotation angle of the video frame.

    Args:
        array: to rotate
        angle: rotation angle in degrees, must be a multiple of 90

    Returns: rotated array

    """
    if angle % 90 != 0:
        raise ValueError(
            f"Rotation angle must be multiple of 90 degrees, but is {angle}"
        )
    return rot90(array, angle // 90)
