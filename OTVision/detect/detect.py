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


from pathlib import Path
from typing import Union

import torch

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files, write_json
from OTVision.helpers.log import log, reset_debug, set_debug

from . import yolo


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
        detections_file = video_file.with_suffix(CONFIG["DEFAULT_FILETYPE"]["DETECT"])

        if not overwrite and detections_file.is_file():
            log.warning(
                f"{detections_file} already exists. To overwrite, set overwrite to True"
            )
            continue

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
        log.info(f"Successfully detected {video_file}")

        write_json(
            dict_to_write=detections_video,
            file=detections_file,
            filetype=CONFIG["DEFAULT_FILETYPE"]["DETECT"],
            overwrite=overwrite,
        )

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
