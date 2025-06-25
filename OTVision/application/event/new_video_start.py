from dataclasses import dataclass


@dataclass(frozen=True)
class NewVideoStartEvent:
    output: str
    width: int
    height: int
    fps: float
