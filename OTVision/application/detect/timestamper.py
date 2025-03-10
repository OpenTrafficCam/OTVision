from abc import ABC, abstractmethod

from OTVision.domain.frame import Frame


class Timestamper(ABC):
    """Interface for adding timestamps to frame data.

    This class defines the interface for timestamp processors that convert raw frame
    dictionaries into Frame objects with proper timestamp information.
    """

    @abstractmethod
    def stamp(self, frame: dict) -> Frame:
        """Add timestamp information to a frame.

        Args:
            frame (dict): Raw frame data dictionary to be processed.

        Returns:
            Frame: A Frame object with added timestamp information.

        """

        raise NotImplementedError
