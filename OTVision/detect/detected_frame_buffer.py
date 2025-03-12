from dataclasses import dataclass
from datetime import datetime, timedelta

from OTVision.application.buffer import Buffer
from OTVision.domain.detection import DetectedFrame


@dataclass
class SourceMetadata:
    source: str
    duration: timedelta
    height: int
    width: int
    fps: float
    start_time: datetime


@dataclass
class FlushEvent:
    source_metadata: SourceMetadata

    @staticmethod
    def create(
        source: str,
        duration: timedelta,
        source_height: int,
        source_width: int,
        source_fps: float,
        start_time: datetime,
    ) -> "FlushEvent":
        return FlushEvent(
            SourceMetadata(
                source,
                duration,
                source_height,
                source_width,
                source_fps,
                start_time=start_time,
            )
        )


@dataclass
class DetectedFrameBufferEvent:
    source_metadata: SourceMetadata
    frames: list[DetectedFrame]


class DetectedFrameBuffer(Buffer[DetectedFrame, DetectedFrameBufferEvent, FlushEvent]):
    def _notify_observers(
        self, elements: list[DetectedFrame], event: FlushEvent
    ) -> None:
        self._subject.notify(
            DetectedFrameBufferEvent(
                source_metadata=event.source_metadata, frames=elements
            )
        )
