"""Video frame provider for loading frames from video files.

This module provides functionality to load frames from video files on-demand,
which is required for appearance-based BOXMOT trackers when processing OTDET
files (file-based tracking).
"""

import logging
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, Protocol

import av
import numpy as np
from numpy import ndarray

from OTVision.dataformat import FILENAME, FILETYPE, VIDEO

logger = logging.getLogger(__name__)


class VideoFrameProvider(Protocol):
    """Protocol for video frame providers.

    Defines the interface for loading frames from video files.
    """

    def get_frame(self, frame_no: int) -> ndarray | None:
        """Get a frame by frame number.

        Args:
            frame_no: 1-indexed frame number to retrieve.

        Returns:
            Frame image as numpy array (BGR format), or None if frame not found.
        """
        ...

    def close(self) -> None:
        """Release video resources."""
        ...


class PyAvVideoFrameProvider:
    """PyAV-based video frame provider with seek-based frame access.

    Opens a video file and provides random access to frames using seeking.

    Attributes:
        _video_path: Path to the video file.
        _container: PyAV container for video access.
        _stream: Video stream from the container.
        _time_base: Time base of the video stream for seeking.
        _fps: Frames per second of the video.
    """

    def __init__(self, video_path: Path) -> None:
        """Initialize the video frame provider.

        Args:
            video_path: Path to the video file.

        Raises:
            FileNotFoundError: If the video file does not exist.
            RuntimeError: If the video file cannot be opened.
        """
        if not video_path.exists():
            raise FileNotFoundError(
                f"Video file not found: {video_path}. "
                "Appearance-based trackers require the original video file "
                "to be available alongside the OTDET file."
            )

        self._video_path = video_path

        try:
            self._container = av.open(str(video_path.absolute()))
            self._container.streams.video[0].thread_type = "AUTO"
            self._stream = self._container.streams.video[0]

            # Get time_base with fallback
            time_base = self._stream.time_base
            if time_base is None:
                raise RuntimeError(f"Video stream has no time_base: {video_path}")
            self._time_base: Fraction = time_base

            # Get fps with fallback
            fps_rate = self._stream.average_rate or self._stream.guessed_rate
            if fps_rate is None:
                raise RuntimeError(f"Could not determine FPS for video: {video_path}")
            self._fps = float(fps_rate)

            self._current_frame_no = 0
        except Exception as e:
            raise RuntimeError(
                f"Failed to open video file: {video_path}. Error: {e}"
            ) from e

        logger.debug(f"Opened video for frame loading: {video_path} (fps={self._fps})")

    def get_frame(self, frame_no: int) -> ndarray | None:
        """Get a frame by frame number using seeking.

        Args:
            frame_no: 1-indexed frame number to retrieve.

        Returns:
            Frame image as numpy array (BGR format), or None if frame not found.
        """
        if frame_no < 1:
            logger.warning(f"Invalid frame number {frame_no}, must be >= 1")
            return None

        try:
            # Calculate timestamp to seek to
            # PyAV uses pts (presentation timestamp), we need to convert frame_no
            # Seek to slightly before the target frame to ensure we can decode it
            target_pts = int((frame_no - 1) / self._fps / self._time_base)

            # Seek to keyframe before target
            self._container.seek(target_pts, stream=self._stream)

            # Decode frames until we reach the target frame
            for frame in self._container.decode(video=0):
                # Calculate which frame number this is
                if frame.pts is not None:
                    current_frame = int(frame.pts * self._time_base * self._fps) + 1
                else:
                    current_frame = frame_no  # Best guess if pts not available

                if current_frame >= frame_no:
                    # Convert to numpy array in BGR format (OpenCV compatible)
                    return frame.to_ndarray(format="bgr24")

            logger.warning(f"Frame {frame_no} not found in video {self._video_path}")
            return None

        except Exception as e:
            logger.error(f"Error getting frame {frame_no} from {self._video_path}: {e}")
            return None

    def close(self) -> None:
        """Release video resources."""
        if hasattr(self, "_container") and self._container is not None:
            self._container.close()
            logger.debug(f"Closed video: {self._video_path}")


class SequentialVideoFrameProvider:
    """Sequential video frame provider for efficient sequential access.

    This provider iterates through frames sequentially, which is more efficient
    when frames are accessed in order (as in tracking). It maintains state to
    avoid re-seeking for sequential frame requests.

    Attributes:
        _video_path: Path to the video file.
        _container: PyAV container for video access.
        _frame_iterator: Iterator over decoded frames.
        _current_frame_no: Current frame number in the iteration.
        _current_frame: Current decoded frame.
    """

    def __init__(self, video_path: Path) -> None:
        """Initialize the sequential video frame provider.

        Args:
            video_path: Path to the video file.

        Raises:
            FileNotFoundError: If the video file does not exist.
            RuntimeError: If the video file cannot be opened.
        """
        if not video_path.exists():
            raise FileNotFoundError(
                f"Video file not found: {video_path}. "
                "Appearance-based trackers require the original video file "
                "to be available alongside the OTDET file."
            )

        self._video_path = video_path
        self._container: av.container.InputContainer | None = None
        self._frame_iterator: Iterator[Any] | None = None
        self._current_frame_no = 0
        self._current_frame: np.ndarray | None = None

        self._open_video()

        logger.debug(f"Opened video for sequential frame loading: {video_path}")

    def _open_video(self) -> None:
        """Open video and initialize frame iterator."""
        try:
            self._container = av.open(str(self._video_path.absolute()))
            self._container.streams.video[0].thread_type = "AUTO"
            self._frame_iterator = self._container.decode(video=0)
            self._current_frame_no = 0
            self._current_frame = None
        except Exception as e:
            raise RuntimeError(
                f"Failed to open video file: {self._video_path}. Error: {e}"
            ) from e

    def get_frame(self, frame_no: int) -> ndarray | None:
        """Get a frame by frame number.

        For sequential access, this is efficient as it advances the iterator.
        For non-sequential access, it may need to re-open the video and seek.

        Args:
            frame_no: 1-indexed frame number to retrieve.

        Returns:
            Frame image as numpy array (BGR format), or None if frame not found.
        """
        if frame_no < 1:
            logger.warning(f"Invalid frame number {frame_no}, must be >= 1")
            return None

        # If requested frame is before current position, we need to restart
        if frame_no < self._current_frame_no:
            logger.debug(
                f"Frame {frame_no} requested but at {self._current_frame_no}, "
                "restarting video"
            )
            self.close()
            self._open_video()

        # Advance to requested frame
        try:
            while self._current_frame_no < frame_no:
                if self._frame_iterator is None:
                    return None
                frame = next(self._frame_iterator)
                self._current_frame_no += 1
                if self._current_frame_no == frame_no:
                    return frame.to_ndarray(format="bgr24")

            # If we're already at the requested frame, return cached result
            if self._current_frame_no == frame_no and self._current_frame is not None:
                return self._current_frame

            return None

        except StopIteration:
            logger.warning(
                f"Reached end of video before frame {frame_no} "
                f"(total frames: {self._current_frame_no})"
            )
            return None
        except Exception as e:
            logger.error(f"Error getting frame {frame_no} from {self._video_path}: {e}")
            return None

    def close(self) -> None:
        """Release video resources."""
        if self._container is not None:
            self._container.close()
            self._container = None
            self._frame_iterator = None
            logger.debug(f"Closed video: {self._video_path}")


def resolve_video_path_from_otdet(otdet_file: Path, metadata: dict) -> Path:
    """Resolve the video file path from OTDET metadata.

    Reconstructs the video file path based on the OTDET file location
    and video metadata stored in the OTDET file.

    Args:
        otdet_file: Path to the OTDET file.
        metadata: OTDET metadata dictionary containing video information.

    Returns:
        Path to the video file.

    Raises:
        KeyError: If required metadata fields are missing.
    """
    video_info = metadata[VIDEO]
    filename = video_info[FILENAME]
    filetype = video_info[FILETYPE]

    # Video file should be in the same directory as the OTDET file
    video_path = otdet_file.parent / f"{filename}{filetype}"

    return video_path
