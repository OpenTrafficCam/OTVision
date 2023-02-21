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


import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import torch
from moviepy.video.io.VideoFileClip import VideoFileClip

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files, write_json
from OTVision.helpers.log import log, reset_debug, set_debug
from OTVision.track.preprocess import DATE_FORMAT, OCCURRENCE

from . import yolo

FILE_NAME_PATTERN = (
    r"(?P<prefix>[A-Za-z0-9]+)"
    r"_FR(?P<frame_rate>\d+)"
    r"_(?P<start_date>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\..*"
)


def main(
    paths: list[Path],
    filetypes: list[str] = CONFIG["FILETYPES"]["VID"],
    model: Union[torch.nn.Module, None] = None,
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
    overwrite: bool = CONFIG["DETECT"]["OVERWRITE"],
    debug: bool = CONFIG["DETECT"]["DEBUG"],
    half_precision: bool = CONFIG["DETECT"]["HALF_PRECISION"],
    force_reload_torch_hub_cache: bool = CONFIG["DETECT"][
        "FORCE_RELOAD_TORCH_HUB_CACHE"
    ],
) -> None:
    """Detects objects in multiple videos and/or images.
    Writes detections to one file per video/object.

    Args:
        paths (list[Path]): List of paths to video files.
        filetypes (list[str], optional): Types of video/image files to be detected.
            Defaults to CONFIG["FILETYPES"]["VID"].
        model (torch.nn.Module, optional): YOLOv5 detection model.
            Defaults to None.
        weights (str, optional): (Pre-)trained weights for YOLOv5 detection model.
            Defaults to CONFIG["DETECT"]["YOLO"]["WEIGHTS"].
        conf (float, optional): YOLOv5 minimum confidence threshold
            for detecting objects. Defaults to CONFIG["DETECT"]["YOLO"]["CONF"].
        iou (float, optional): YOLOv5 IOU threshold for detecting objects.
            Defaults to CONFIG["DETECT"]["YOLO"]["IOU"].
        size (int, optional): YOLOv5 image size.
            Defaults to CONFIG["DETECT"]["YOLO"]["IMGSIZE"].
        chunksize (int, optional): YOLOv5 chunksize.
            Defaults to CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"].
        normalized (bool, optional): Whether or not to normalize detections
            to image dimensions. Defaults to CONFIG["DETECT"]["YOLO"]["NORMALIZED"].
        overwrite (bool, optional): Whether or not to overwrite
            existing detections files. Defaults to CONFIG["DETECT"]["OVERWRITE"].
        half_precision (bool, optional): Whether to use half precision (FP16) for
            inference speed up. Only works for gpu.
            Defaults to CONFIG["DETECT"]["HALF_PRECISION"].
        force_reload_torch_hub_cache (bool, optional): Whether to force reload torch
            hub cache. Defaults to CONFIG["DETECT"]["FORCE_RELOAD_TORCH_HUB_CACHE].
        debug (bool, optional): Whether or not logging in debug mode.
            Defaults to CONFIG["DETECT"]["DEBUG"].
    """
    log.info("Start detection")
    if debug:
        set_debug()

    video_files = get_files(paths=paths, filetypes=filetypes)

    if not video_files:
        raise FileNotFoundError(f"No videos of type '{filetypes}' found to detect!")

    if not model:
        yolo_model = yolo.loadmodel(
            weights,
            conf,
            iou,
            force_reload=force_reload_torch_hub_cache,
            half_precision=half_precision,
        )
    else:
        yolo_model = (
            model.half() if torch.cuda.is_available() and half_precision else model
        )
        yolo_model.conf = conf
        yolo_model.iou = iou
    log.info("Model prepared")

    for video_file in video_files:
        log.info(f"Try detecting {video_file}")
        detections_video = yolo.detect_video(
            file=video_file,
            model=yolo_model,
            weights=weights,
            conf=conf,
            iou=iou,
            size=size,
            chunksize=chunksize,
            normalized=normalized,
        )
        log.info("Video detected")
        stamped_detections = add_timestamps(detections_video, video_file)
        write(stamped_detections, video_file, overwrite=overwrite)

    if debug:
        reset_debug()
    return None


class FormatNotSupportedError(Exception):
    pass


def _create_chunks(files: list[Path], chunksize: int) -> list[list[Path]]:
    """Splits list in several lists of certain chunksize.

    Args:
        files (list[Path]): Full list.
        chunksize (int): Chunksize to split list into.

    Returns:
        list[list[Path]]: list of lists of certain chunksize.
    """
    if chunksize == 0:
        return [files]
    chunk_starts = range(0, len(files), chunksize)
    return [files[i : i + chunksize] for i in chunk_starts]


def add_timestamps(detections: dict, video_file: Path) -> dict:
    return Timestamper().stamp(detections, video_file)


class InproperFormattedFilename(Exception):
    pass


class Timestamper:
    def stamp(self, detections: dict, video_file: Path) -> dict:
        """This method adds timestamps when the frame occurred in real time to each
        frame.

        Args:
            detections (dict): dictionary containing all frames
            video_file (Path): path to video file

        Returns:
            dict: input dictionary with additional occurrence per frame
        """
        start_time = self._get_start_time_from(video_file)
        time_per_frame = self._get_time_per_frame(detections, video_file)
        return self._stamp(detections, start_time, time_per_frame)

    def _get_start_time_from(self, video_file: Path) -> datetime:
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
            start_date: str = match.group("start_date")
            return datetime.strptime(start_date, "%Y-%m-%d_%H-%M-%S")
        raise InproperFormattedFilename(f"Could not parse {video_file.name}.")

    def _get_time_per_frame(self, detections: dict, video_file: Path) -> timedelta:
        """Calculates the duration for each frame. This is done using the total
        duration of the video and the number of frames.

        Args:
            detections (dict): dictionary containing all frames
            video_file (Path): path to video file

        Returns:
            timedelta: duration per frame
        """
        duration = self._get_duration(video_file)
        number_of_frames = len(detections["data"].keys())
        return duration / number_of_frames

    def _get_duration(self, video_file: Path) -> timedelta:
        """Get the duration of the video

        Args:
            video_file (Path): path to video file

        Returns:
            timedelta: duration of the video
        """
        clip = VideoFileClip(str(video_file.absolute()))
        return timedelta(seconds=clip.duration)

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
        data: dict = detections["data"]
        for key, value in data.items():
            occurrence = start_date + (int(key) - 1) * time_per_frame
            value[OCCURRENCE] = occurrence.strftime(DATE_FORMAT)
        return detections


def write(
    detections: dict,  # TODO: Type hint nested dict during refactoring"
    img_or_video_file: Path,
    overwrite: bool = CONFIG["DETECT"]["OVERWRITE"],
) -> None:
    """Writes detections of a video or image to a json-like file.

    Args:
        detections (dict): Detections of a video or image.
        img_or_video_file (Path): Path to image or video of detections.
        overwrite (bool, optional): Wheter or not to overwrite existing detections file.
            Defaults to CONFIG["DETECT"]["OVERWRITE"].
    """
    # ?: Check overwrite before detecting instead of before writing detections?
    filetype = CONFIG["DEFAULT_FILETYPE"]["DETECT"]
    detection_file = Path(img_or_video_file).with_suffix(filetype)
    detections_file_already_exists = detection_file.is_file()
    if overwrite or not detections_file_already_exists:
        # Write JSON
        t_json_start = time.perf_counter()
        write_json(detections, detection_file, filetype=filetype, overwrite=overwrite)
        t_json_end = time.perf_counter()
        log.info(f"Writing .otdet took: {t_json_end - t_json_start:0.4f}s")
        if detections_file_already_exists:
            log.info(f"{detection_file} overwritten")
        else:
            log.info(f"{detection_file} written")
    else:
        log.info(f"{detection_file} already exists. To overwrite, set overwrite=True")
