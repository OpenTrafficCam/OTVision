from dataclasses import dataclass


@dataclass(frozen=True, repr=True)
class Detection:
    label: str
    conf: float
    x: float
    y: float
    w: float
    h: float
