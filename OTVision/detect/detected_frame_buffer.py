from dataclasses import dataclass
from datetime import datetime, timedelta

from OTVision.abstraction.observer import AsyncObservable, AsyncSubject
from OTVision.application.buffer import Buffer
from OTVision.domain.frame import DetectedFrame


@dataclass
class SourceMetadata:
    source: str
    output: str
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
        output: str,
        duration: timedelta,
        source_height: int,
        source_width: int,
        source_fps: float,
        start_time: datetime,
    ) -> "FlushEvent":
        return FlushEvent(
            SourceMetadata(
                source,
                output,
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


class DetectedFrameBuffer(
    Buffer[DetectedFrame, FlushEvent], AsyncObservable[DetectedFrameBufferEvent]
):
    def __init__(self, subject: AsyncSubject[DetectedFrameBufferEvent]) -> None:
        Buffer.__init__(self)
        AsyncObservable.__init__(self, subject)

    async def on_flush(self, event: FlushEvent) -> None:
        buffered_elements = self._get_buffered_elements()
        await self._notify_observers(buffered_elements, event)
        self._reset_buffer()

    async def _notify_observers(
        self, elements: list[DetectedFrame], event: FlushEvent
    ) -> None:
        await self._subject.notify(
            DetectedFrameBufferEvent(
                source_metadata=event.source_metadata, frames=elements
            )
        )

    async def buffer(self, to_buffer: DetectedFrame) -> None:
        self._buffer.append(to_buffer.without_image())
