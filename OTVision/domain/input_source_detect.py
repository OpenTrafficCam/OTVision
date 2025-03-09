from abc import ABC, abstractmethod
from typing import Generator

from OTVision.abstraction.observer import Observable
from OTVision.detect.detected_frame_buffer import FlushEvent
from OTVision.domain.frame import Frame


class InputSourceDetect(Observable[FlushEvent], ABC):
    """Interface for input sources that generate frames and notify about flush events.

    This class combines the Observable pattern for flush events with frame generation
    capabilities. It serves as a base for components that read frames from various
    sources (e.g., video files, camera feeds) and need to notify observers about
    buffer flush events during processing.
    """

    @abstractmethod
    def produce(self) -> Generator[Frame, None, None]:
        """Generate a stream of frames from the input source.

        Implementations should yield Frame objects one at a time from the source,
        while potentially triggering flush events through the Observable interface
        at appropriate points (e.g., end of video segments or buffer boundaries).

        Returns:
            Generator[Frame, None, None]: A generator yielding Frame objects
                sequentially from the input source.
        """

        raise NotImplementedError
