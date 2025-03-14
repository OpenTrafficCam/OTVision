import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from OTVision.application.detect.timestamper import Timestamper
from OTVision.application.frame_count_provider import FrameCountProvider
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.config import DATETIME_FORMAT
from OTVision.dataformat import FRAME
from OTVision.domain.frame import Frame, FrameKeys
from OTVision.helpers.date import parse_date_string_to_utc_datime
from OTVision.helpers.files import (
    FILE_NAME_PATTERN,
    START_DATE,
    InproperFormattedFilename,
)
from OTVision.helpers.video import get_duration


class VideoTimestamper(Timestamper):
    """Add timestamps when the frame occurred in real time to each frame.


    Args:
        video_file (Path): Path to the video file being processed.
        expected_duration (timedelta | None): expected duration of the video used to
            calculate the number of actual frames per second.
        frame_count_provider (FrameCountProvider): Provider for the total number of
            frames in a video file.
    """

    def __init__(
        self,
        video_file: Path,
        expected_duration: timedelta | None,
        frame_count_provider: FrameCountProvider,
        start_time: datetime | None,
    ) -> None:
        self._video_file = video_file
        self._expected_duration = expected_duration
        self._start_time = start_time

        self._number_of_frames = frame_count_provider.provide(self._video_file)
        self._time_per_frame = self._get_time_per_frame()

    def stamp(self, frame: dict) -> Frame:
        """This method adds timestamps when the frame occurred in real time to each
        frame.

        Returns:
            Frame: frame with occurrence.
        """
        start_time = parse_start_time_from(
            self._video_file, start_time=self._start_time
        )
        frame_number = frame[FRAME]
        # Frame numbers start from 1
        occurrence = start_time + (frame_number - 1) * self._time_per_frame

        return Frame(
            data=frame[FrameKeys.data],
            frame=frame[FrameKeys.frame],
            source=frame[FrameKeys.source],
            occurrence=occurrence,
        )

    def _get_time_per_frame(self) -> timedelta:
        """Calculates the duration for each frame. This is done using the total
        duration of the video and the number of frames.

        Returns:
            timedelta: duration per frame
        """
        if self._expected_duration:
            duration = self._expected_duration
        else:
            duration = get_duration(self._video_file)

        return duration / self._number_of_frames


class TimestamperFactory:
    """
    Factory class for creating timestamper instances.

    This class provides methods to create and configure instances of the
    Timestamper class or its derivatives.
    """

    def __init__(
        self,
        frame_count_provider: FrameCountProvider,
        get_current_config: GetCurrentConfig,
    ) -> None:
        self._frame_count_provider = frame_count_provider
        self._get_current_config = get_current_config

    def create_video_timestamper(
        self, video_file: Path, expected_duration: timedelta | None
    ) -> Timestamper:
        """Creates a video timestamper to handle video timestamping operations.

        This function initializes a Timestamper object for the given video file,
        allowing timestamp metadata to be processed or manipulated based on
        the provided duration and frame count.

        Args:
            video_file (Path): the video file of frames to be timestamped.
            expected_duration (timedelta | None): specifies the expected duration of the
                video. If None, the duration is not pre-defined.

        Returns:
            Timestamper: An instance of Timestamper tailored to the provided video
                file and parameters.

        """
        return VideoTimestamper(
            video_file=video_file,
            expected_duration=expected_duration,
            frame_count_provider=self._frame_count_provider,
            start_time=self._get_current_config.get().detect.start_time,
        )


def parse_start_time_from(video_file: Path, start_time: datetime | None) -> datetime:
    """Parse the given filename and retrieve the start date of the video.

    Args:
        video_file (Path): path to video file
        start_time (datetime | None): start time of the video. The start time will be
            parsed from the video file name if  no start time is provided.

    Raises:
        InproperFormattedFilename: if the filename is not formatted as expected, an
        exception will be raised

    Returns:
        datetime: start date of the video
    """
    if start_time is not None:
        return start_time
    match = re.search(
        FILE_NAME_PATTERN,
        video_file.name,
    )
    if match:
        start_date: str = match.group(START_DATE)
        return parse_date_string_to_utc_datime(start_date, DATETIME_FORMAT).replace(
            tzinfo=timezone.utc
        )

    raise InproperFormattedFilename(f"Could not parse {video_file.name}.")
