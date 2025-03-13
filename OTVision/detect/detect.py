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
from datetime import datetime, timezone
from pathlib import Path

from OTVision.config import DATETIME_FORMAT
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

log = logging.getLogger(LOGGER_NAME)


class OTVisionVideoDetect(DetectedFrameConsumer):
    def __init__(self, producer: DetectedFrameProducer) -> None:
        self._producer = producer

    def start(self) -> None:
        """Starts the detection of objects in multiple videos and/or images."""
        self.consume()

    def consume(self) -> None:
        for _ in self._producer.produce():
            pass


def parse_start_time_from(video_file: Path, start_time: datetime | None) -> datetime:
    """Parse the given filename and retrieve the start date of the video.

    Args:
        video_file (Path): path to video file

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
