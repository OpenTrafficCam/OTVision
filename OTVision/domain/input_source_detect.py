from abc import ABC, abstractmethod
from typing import Iterator

from OTVision.domain.frame import Frame


class InputSourceDetect(ABC):
    """Interface for input sources that generate frames and notify about flush events.

    This class combines the Observable pattern for flush events with frame generation
    capabilities. It serves as a base for components that read frames from various
    sources (e.g., video files, camera feeds) and need to notify observers about
    buffer flush events during processing.
    """

    @abstractmethod
    def produce(self) -> Iterator[Frame]:
        """Generate a stream of frames from the input source.

        Implementations should yield Frame objects one at a time from the source,
        while potentially triggering flush events through the Observable interface
        at appropriate points (e.g., end of video segments or buffer boundaries).

        Returns:
            Iterator [Frame]: A generator yielding Frame objects
                sequentially from the input source.
        """

        raise NotImplementedError
