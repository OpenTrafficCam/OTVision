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


import time
from pathlib import Path
from typing import Union

import torch
import ujson

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files, has_filetype
from OTVision.helpers.log import log, reset_debug, set_debug

from . import yolo


def main(
    paths: list[Path],
    filetypes: list[str] = CONFIG["FILETYPES"]["VID_IMG"],
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

    files = get_files(paths=paths, filetypes=filetypes)
    video_files, img_files = _split_to_video_img_paths(files)
    log.info("Files splitted in videos and images")

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
        write(detections_video, video_file, overwrite=overwrite)

    if debug:
        reset_debug()
    return None


def _split_to_video_img_paths(
    files: list[Path],
    video_formats: list[str] = CONFIG["FILETYPES"]["VID"],
    img_formats: list[str] = CONFIG["FILETYPES"]["IMG"],
) -> tuple[list[Path], list[Path]]:
    """
    Divides a list of files in video files and image files.

    Args:
        files (list[Path]): List of video and/or image file paths.
        video_formats (list[str], optional): _description_.
            Defaults to CONFIG["FILETYPES"]["VID"].
        img_formats (list[str], optional): _description_.
            Defaults to CONFIG["FILETYPES"]["IMG"].

    Raises:
        FormatNotSupportedError: If format of a path is not in
            video_formats or img_formats.

    Returns:
        tuple[list[Path], list[Path]]: List of video paths and list of image paths
    """

    video_files, img_files = [], []
    for file in files:
        if has_filetype(file, video_formats):
            video_files.append(file)
        elif has_filetype(file, img_formats):
            img_files.append(file)
        else:
            raise FormatNotSupportedError(
                f"The format of path is not supported ({file})"
            )
    return video_files, img_files


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
    detection_file = Path(img_or_video_file).with_suffix(
        CONFIG["DEFAULT_FILETYPE"]["DETECT"]
    )
    detections_file_already_exists = detection_file.is_file()
    if overwrite or not detections_file_already_exists:
        # Write JSON
        with open(detection_file, "w") as f:
            t_json_start = time.perf_counter()
            ujson.dump(detections, f, indent=4)
            t_json_end = time.perf_counter()
            log.info(f"Writing .otdet took: {t_json_end - t_json_start:0.4f}s")
        if detections_file_already_exists:
            log.info(f"{detection_file} overwritten")
        else:
            log.info(f"{detection_file} written")
    else:
        log.info(f"{detection_file} already exists. To overwrite, set overwrite=True")
