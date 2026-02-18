from abc import ABC, abstractmethod
from typing import AsyncIterator

from OTVision.domain.frame import DetectedFrame


class DetectedFrameConsumer(ABC):
    """Interface for components that consume detected frames."""

    @abstractmethod
    async def consume(self) -> None:
        """Consume detected frames."""
        raise NotImplementedError


class DetectedFrameProducer(ABC):
    """Interface for components that generate detected frames.

    This class defines the interface for components that create or provide
    a stream of detected frames for further processing.
    """

    @abstractmethod
    def produce(self) -> AsyncIterator[DetectedFrame]:
        """Generate a stream of detected frames.

        Returns:
            AsyncIterator[DetectedFrame ]: A stream of detected frames.
        """
        ...
