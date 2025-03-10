from abc import ABC
from pathlib import Path


class FrameCountProvider(ABC):
    """Abstract base class for retrieving the total number of frames in a video file.

    This interface defines the contract for components that can determine the frame
    count of video files. Implementations might use different methods or libraries
    to calculate the total number of frames.
    """

    def provide(self, video_file: Path) -> int:
        """Get the total number of frames in a video file.

        Args:
            video_file (Path): Path to the video file to analyze.

        Returns:
            int: Total number of frames in the video.

        """
        raise NotImplementedError
