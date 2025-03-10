"""
OTVision main module to detect objects in single or multiple images or videos.
"""

# Copyright (C) 2022 OpenTrafficCam Contributors
# <https://github.com/OpenTrafficCam
# <team@opentrafficcam.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from OTVision.dataformat import DATA, LENGTH, METADATA, RECORDED_START_DATE, VIDEO
from OTVision.domain.detect_producer_consumer import (
    DetectedFrameConsumer,
    DetectedFrameProducer,
)
from OTVision.helpers.date import parse_date_string_to_utc_datime
from OTVision.helpers.files import (
    FILE_NAME_PATTERN,
    START_DATE,
    InproperFormattedFilename,
)
from OTVision.helpers.log import LOGGER_NAME
from OTVision.helpers.video import get_duration
from OTVision.track.preprocess import OCCURRENCE

log = logging.getLogger(LOGGER_NAME)
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"


class OTVisionVideoDetect(DetectedFrameConsumer):
    def __init__(self, producer: DetectedFrameProducer) -> None:
        self._producer = producer

    def start(self) -> None:
        """Starts the detection of objects in multiple videos and/or images."""
        self.consume()

    def consume(self) -> None:
        for _ in self._producer.produce():
            pass


def add_timestamps(
    detections: dict, video_file: Path, expected_duration: timedelta | None
) -> dict:
    return Timestamper().stamp(detections, video_file, expected_duration)


class Timestamper:
    def stamp(
        self, detections: dict, video_file: Path, expected_duration: timedelta | None
    ) -> dict:
        """This method adds timestamps when the frame occurred in real time to each
        frame.

        Args:
            detections (dict): dictionary containing all frames
            video_file (Path): path to video file
            expected_duration (timedelta | None): expected duration of the video used to
                calculate the number of actual frames per second

        Returns:
            dict: input dictionary with additional occurrence per frame
        """
        start_time = parse_start_time_from(video_file)
        actual_duration = get_duration(video_file)
        if expected_duration:
            time_per_frame = self._get_time_per_frame(detections, expected_duration)
        else:
            time_per_frame = self._get_time_per_frame(detections, actual_duration)
        self._update_metadata(detections, start_time, actual_duration)
        return self._stamp(detections, start_time, time_per_frame)

    @staticmethod
    def _get_time_per_frame(detections: dict, duration: timedelta) -> timedelta:
        """Calculates the duration for each frame. This is done using the total
        duration of the video and the number of frames.

        Args:
            detections (dict): dictionary containing all frames
            video_file (Path): path to video file

        Returns:
            timedelta: duration per frame
        """
        number_of_frames = len(detections[DATA].keys())
        return duration / number_of_frames

    @staticmethod
    def _update_metadata(
        detections: dict, start_time: datetime, duration: timedelta
    ) -> dict:
        detections[METADATA][VIDEO][RECORDED_START_DATE] = start_time.timestamp()
        detections[METADATA][VIDEO][LENGTH] = str(duration)
        return detections

    def _stamp(
        self, detections: dict, start_date: datetime, time_per_frame: timedelta
    ) -> dict:
        """Add a timestamp (occurrence in real time) to each frame.

        Args:
            detections (dict): dictionary containing all frames
            start_date (datetime): start date of the video recording
            time_per_frame (timedelta): duration per frame

        Returns:
            dict: dictionary containing all frames with their occurrence in real time
        """
        data: dict = detections[DATA]
        for key, value in data.items():
            occurrence = start_date + (int(key) - 1) * time_per_frame
            value[OCCURRENCE] = occurrence.timestamp()
        return detections


def parse_start_time_from(video_file: Path) -> datetime:
    """Parse the given filename and retrieve the start date of the video.

    Args:
        video_file (Path): path to video file

    Raises:
        InproperFormattedFilename: if the filename is not formatted as expected, an
        exception will be raised

    Returns:
        datetime: start date of the video
    """
    match = re.search(
        FILE_NAME_PATTERN,
        video_file.name,
    )
    if match:
        start_date: str = match.group(START_DATE)
        return parse_date_string_to_utc_datime(start_date, "%Y-%m-%d_%H-%M-%S").replace(
            tzinfo=timezone.utc
        )

    raise InproperFormattedFilename(f"Could not parse {video_file.name}.")
