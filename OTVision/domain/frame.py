from datetime import datetime
from typing import Literal, Optional, TypedDict

from numpy import ndarray


class FrameKeys:
    """Keys to access Frame dictionary."""

    data: Literal["data"] = "data"
    frame: Literal["frame"] = "frame"
    source: Literal["source"] = "source"
    occurrence: Literal["occurrence"] = "occurrence"


class Frame(TypedDict):
    """Frame definition.

    Attributes:
        data (Optional[ndarray]): The frame data as a numpy array, can be None.
        frame (int): The frame number.
        source (str): The source identifier of the frame.
        occurrence (datetime): Timestamp when the frame was captured/created.
    """

    data: Optional[ndarray]
    frame: int
    source: str
    occurrence: datetime
