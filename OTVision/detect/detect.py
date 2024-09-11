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

from OTVision.config import CONFIG, DEFAULT_FILETYPE, DETECT, FILETYPES, OVERWRITE, VID
from OTVision.dataformat import DATA, LENGTH, METADATA, RECORDED_START_DATE, VIDEO
from OTVision.detect.otdet import OtdetBuilder
from OTVision.detect.yolo import Yolov8
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


def main(
    model: Yolov8,
    paths: list[Path],
    expected_duration: timedelta,
    filetypes: list[str] = CONFIG[FILETYPES][VID],
    overwrite: bool = CONFIG[DETECT][OVERWRITE],
) -> None:
    """Detects objects in multiple videos and/or images.
    Writes detections to one file per video/object.

    Args:
        model (Yolov8): YOLOv8 detection model.
        paths (list[Path]): List of paths to video files.
        expected_duration (timedelta): expected duration of the video. All frames are
            evenly spread across this duration
        filetypes (list[str], optional): Types of video/image files to be detected.
            Defaults to CONFIG["FILETYPES"]["VID"].
        overwrite (bool, optional): Whether or not to overwrite
            existing detections files. Defaults to CONFIG["DETECT"]["OVERWRITE"].
    """

    video_files = get_files(paths=paths, filetypes=filetypes)

    start_msg = f"Start detection of {len(video_files)} video files"
    log.info(start_msg)
    print(start_msg)

    if not video_files:
        log.warning(f"No videos of type '{filetypes}' found to detect!")
        return

    for video_file in tqdm(video_files, desc="Detected video files", unit=" files"):
        detections_file = video_file.with_suffix(CONFIG[DEFAULT_FILETYPE][DETECT])

        if not overwrite and detections_file.is_file():
            log.warning(
                f"{detections_file} already exists. To overwrite, set overwrite to True"
            )
            continue

        log.info(f"Detect {video_file}")

        detections = model.detect(file=video_file)

        video_width, video_height = get_video_dimensions(video_file)
        video_fps = get_fps(video_file)
        actual_frames = len(detections)
        actual_fps = actual_frames / expected_duration.total_seconds()
        otdet = OtdetBuilder(
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
        ).build(detections)

        stamped_detections = add_timestamps(otdet, video_file, expected_duration)
        write_json(
            stamped_detections,
            file=detections_file,
            filetype=CONFIG[DEFAULT_FILETYPE][DETECT],
            overwrite=overwrite,
        )

        log.info(f"Successfully detected and wrote {detections_file}")

    finished_msg = "Finished detection"
    log.info(finished_msg)
    print(finished_msg)

    return None


class FormatNotSupportedError(Exception):
    pass


def add_timestamps(
    detections: dict, video_file: Path, expected_duration: timedelta
) -> dict:
    return Timestamper().stamp(detections, video_file, expected_duration)


class Timestamper:
    def stamp(
        self, detections: dict, video_file: Path, expected_duration: timedelta
    ) -> dict:
        """This method adds timestamps when the frame occurred in real time to each
        frame.

        Args:
            detections (dict): dictionary containing all frames
            video_file (Path): path to video file
            expected_duration (timedelta): expected duration of the video used to
                calculate the number of actual frames per second

        Returns:
            dict: input dictionary with additional occurrence per frame
        """
        start_time = self._get_start_time_from(video_file)
        duration = get_duration(video_file)
        time_per_frame = self._get_time_per_frame(detections, expected_duration)
        self._update_metadata(detections, start_time, duration)
        return self._stamp(detections, start_time, time_per_frame)

    @staticmethod
    def _get_start_time_from(video_file: Path) -> datetime:
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
            return parse_date_string_to_utc_datime(
                start_date, "%Y-%m-%d_%H-%M-%S"
            ).replace(tzinfo=timezone.utc)

        raise InproperFormattedFilename(f"Could not parse {video_file.name}.")

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
