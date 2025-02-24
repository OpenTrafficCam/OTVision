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

from tqdm import tqdm

from OTVision.config import Config
from OTVision.dataformat import DATA, LENGTH, METADATA, RECORDED_START_DATE, VIDEO
from OTVision.detect.otdet import OtdetBuilder, OtdetBuilderConfig
from OTVision.detect.yolo import create_model
from OTVision.helpers.date import parse_date_string_to_utc_datime
from OTVision.helpers.files import (
    FILE_NAME_PATTERN,
    START_DATE,
    InproperFormattedFilename,
    get_files,
    write_json,
)
from OTVision.helpers.log import LOGGER_NAME
from OTVision.helpers.video import get_duration, get_fps, get_video_dimensions
from OTVision.track.preprocess import OCCURRENCE

log = logging.getLogger(LOGGER_NAME)
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"


class OTVisionDetect:
    @property
    def config(self) -> Config:
        if self._config is None:
            raise ValueError("Config is missing!")
        return self._config

    def __init__(self, otdet_builder: OtdetBuilder) -> None:
        self._config: Config | None = None
        self._otdet_builder = otdet_builder

    def update_config(self, config: Config) -> None:
        self._config = config

    def start(self) -> None:
        """Starts the detection of objects in multiple videos and/or images.

        Writes detections to one file per video/object.

        """
        filetypes = self.config.filetypes.video_filetypes.to_list()
        video_files = get_files(paths=self.config.detect.paths, filetypes=filetypes)

        start_msg = f"Start detection of {len(video_files)} video files"
        log.info(start_msg)
        print(start_msg)

        if not video_files:
            log.warning(f"No videos of type '{filetypes}' found to detect!")
            return

        model = create_model(
            weights=self.config.detect.yolo_config.weights,
            confidence=self.config.detect.yolo_config.conf,
            iou=self.config.detect.yolo_config.iou,
            img_size=self.config.detect.yolo_config.img_size,
            half_precision=self.config.detect.half_precision,
            normalized=self.config.detect.yolo_config.normalized,
        )
        for video_file in tqdm(video_files, desc="Detected video files", unit=" files"):
            detections_file = derive_filename(
                video_file=video_file,
                detect_start=self.config.detect.detect_start,
                detect_end=self.config.detect.detect_end,
                detect_suffix=self.config.filetypes.detect,
            )

            try:
                parse_start_time_from(video_file)
            except InproperFormattedFilename:
                log.warning(
                    f"Video file name of '{video_file}' must include date "
                    f"and time in format: {DATETIME_FORMAT}"
                )
                continue

            if not self.config.detect.overwrite and detections_file.is_file():
                log.warning(
                    f"{detections_file} already exists. To overwrite, set overwrite "
                    "to True"
                )
                continue

            log.info(f"Detect {video_file}")

            video_fps = get_fps(video_file)
            detect_start_in_frames = convert_seconds_to_frames(
                self.config.detect.detect_start, video_fps
            )
            detect_end_in_frames = convert_seconds_to_frames(
                self.config.detect.detect_end, video_fps
            )
            detections = model.detect(
                file=video_file,
                detect_start=detect_start_in_frames,
                detect_end=detect_end_in_frames,
            )

            video_width, video_height = get_video_dimensions(video_file)
            actual_duration = get_duration(video_file)
            actual_frames = len(detections)
            if (expected_duration := self.config.detect.expected_duration) is not None:
                actual_fps = actual_frames / expected_duration.total_seconds()
            else:
                actual_fps = actual_frames / actual_duration.total_seconds()
            otdet = self._otdet_builder.add_config(
                OtdetBuilderConfig(
                    conf=model.confidence,
                    iou=model.iou,
                    video=video_file,
                    video_width=video_width,
                    video_height=video_height,
                    expected_duration=expected_duration,
                    recorded_fps=video_fps,
                    actual_fps=actual_fps,
                    actual_frames=actual_frames,
                    detection_img_size=model.img_size,
                    normalized=model.normalized,
                    detection_model=model.weights,
                    half_precision=model.half_precision,
                    chunksize=1,
                    classifications=model.classifications,
                    detect_start=self.config.detect.detect_start,
                    detect_end=self.config.detect.detect_end,
                )
            ).build(detections)

            stamped_detections = add_timestamps(otdet, video_file, expected_duration)
            write_json(
                stamped_detections,
                file=detections_file,
                filetype=self.config.filetypes.detect,
                overwrite=self.config.detect.overwrite,
            )

            log.info(f"Successfully detected and wrote {detections_file}")

        finished_msg = "Finished detection"
        log.info(finished_msg)
        print(finished_msg)


def derive_filename(
    video_file: Path,
    detect_suffix: str,
    detect_start: int | None = None,
    detect_end: int | None = None,
) -> Path:
    """
    Generates a filename for detection files by appending specified start and end
    markers and a suffix to the stem of the input video file.

    Args:
        video_file (Path): The input video file whose filename is to be modified.
        detect_start (int | None): The starting marker to append to the filename.
            If None, no starting marker will be appended.
        detect_end (int | None): The ending marker to append to the filename. If None,
            no ending marker will be appended.
        detect_suffix (str): The file suffix to apply to the derived filename.

    Returns:
        Path: The modified video file path with the updated stem and suffix applied.
    """
    cutout = ""
    if detect_start is not None:
        cutout += f"_start_{detect_start}"
    if detect_end is not None:
        cutout += f"_end_{detect_end}"
    new_stem = f"{video_file.stem}{cutout}"
    return video_file.with_stem(new_stem).with_suffix(detect_suffix)


def convert_seconds_to_frames(seconds: int | None, fps: float) -> int | None:
    if seconds is None:
        return None
    return round(seconds * fps)


class FormatNotSupportedError(Exception):
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
