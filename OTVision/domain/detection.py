from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, repr=True)
class Detection:
    label: str
    conf: float
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class DetectedFrame:
    source: str
    frame_number: int
    occurrence: datetime
    detections: list[Detection]
