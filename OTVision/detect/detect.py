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


import json
from pathlib import Path
from typing import Union

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files, is_in_format
from OTVision.helpers.log import log

from . import yolo


def main(
    paths: Union[list, str, Path],
    filetypes: list = CONFIG["FILETYPES"]["VID"],
    model=None,
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
    overwrite: bool = CONFIG["DETECT"]["OVERWRITE"],
    ot_labels_enabled: bool = False,
    debug: bool = CONFIG["DETECT"]["DEBUG"],
):
    log.info("Start detection")
    if debug:
        log.setLevel("DEBUG")
        log.debug("Debug mode on")

    if not model:
        yolo_model = yolo.loadmodel(weights, conf, iou)
    else:
        yolo_model = model
        yolo_model.conf = conf
        yolo_model.iou = iou
    log.info("Model prepared")

    files = get_files(paths=paths, filetypes=filetypes)
    video_files, img_files = _split_to_video_img_paths(files)
    log.info("Files splitted in videos and images")

    for video_file in video_files:
        log.info(f"Try detecting {video_file}")
        detections_video = yolo.detect_video(
            file_path=video_file,
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

    log.info(f"Try detecting {len(img_files)} images")
    img_file_chunks = _create_chunks(img_files, chunksize)
    detections_img_file_chunks = yolo.detect_images(
        file_chunks=img_file_chunks,
        model=yolo_model,
        weights=weights,
        conf=conf,
        iou=iou,
        size=size,
        chunksize=chunksize,
        normalized=normalized,
        ot_labels_enabled=ot_labels_enabled,
    )
    log.info("Images detected")

    if ot_labels_enabled:
        return detections_img_file_chunks
    for img_file, detection in zip(img_files, detections_img_file_chunks):
        write(detection, img_file)


def _split_to_video_img_paths(
    files,
    video_formats=CONFIG["FILETYPES"]["VID"],
    img_formats=CONFIG["FILETYPES"]["IMG"],
):
    """Divide a list of files in video files and other files.

    Args:
        file_paths (list): The list of files.
        vidoe_formats

    Returns:
        [list(str), list{str)] : list of video paths and list of images paths
    """
    video_files, img_files = [], []
    for file in files:
        if is_in_format(file, video_formats):
            video_files.append(file)
        elif is_in_format(file, img_formats):
            img_files.append(file)
        else:
            raise FormatNotSupportedError(
                f"The format of path is not supported ({file})"
            )
    return video_files, img_files


class FormatNotSupportedError(Exception):
    pass


def _create_chunks(files, chunksize):
    if chunksize == 0:
        return files
    chunk_starts = range(0, len(files), chunksize)
    return [files[i : i + chunksize] for i in chunk_starts]


def write(detections, img_or_video_file, overwrite=CONFIG["DETECT"]["OVERWRITE"]):
    # ?: Check overwrite before detecting instead of before writing detections?
    detection_file = Path(img_or_video_file).with_suffix(CONFIG["DEFAULT_FILETYPE"]["DETECT"])
    detections_file_already_exists = detection_file.is_file()
    if overwrite or not detections_file_already_exists:
        # Write JSON
        with open(detection_file, "w") as f:
            json.dump(detections, f, indent=4)
        if detections_file_already_exists:
            log.info(f"{detection_file} overwritten")
        else:
            log.info(f"{detection_file} written")
    else:
        log.info(
            f"{detection_file} already exists. To overwrite, set overwrite=True"
        )
