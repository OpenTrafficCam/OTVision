"""
Module to call yolov5/detect.py with arguments
"""

# Copyright (C) 2020 OpenTrafficCam Contributors
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

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files, is_in_format

from . import yolo
import os

# def main(paths, filetypes, det_config={}):
#     files = get_files(paths, filetypes)
#     multiple_videos(files, **det_config)

# TODO: Add option to allow or prevent overwrite in detect


def main(
    files,
    filetypes: list = CONFIG["FILETYPES"]["VID"],
    model=None,
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
    ot_labels_enabled: bool = False,
):  # sourcery skip: merge-dict-assign
    if model is None:
        yolo_model = yolo.loadmodel(weights, conf, iou)
    else:
        yolo_model = model
        yolo_model.conf = conf
        yolo_model.iou = iou

    file_paths = get_files(paths=files, filetypes=filetypes)

    # split file paths to two groups -> videos | images
    # only when accepting multiple filetypes
    video_paths, frame_paths = _split_to_video_img_paths(file_paths)

    frame_chunks = _create_chunks(frame_paths, chunksize)

    for path in video_paths:
        detections_videos = yolo.detect_video(
            file_path=path,
            model=yolo_model,
            weights=weights,
            conf=conf,
            iou=iou,
            size=size,
            chunksize=chunksize,
            normalized=normalized,
        )
        save_detections(detections_videos, path)

    detections_chunks = yolo.detect_images(
        file_chunks=frame_chunks,
        model=yolo_model,
        weights=weights,
        conf=conf,
        iou=iou,
        size=size,
        chunksize=chunksize,
        normalized=normalized,
        ot_labels_enabled=ot_labels_enabled,
    )
    # TODO: what happens if no frames detected
    # save detection information to corresponding frame path
    if ot_labels_enabled:
        return detections_chunks
    else:
        for frame_path, detection in zip(frame_paths, detections_chunks):
            save_detections(detection, frame_path)


def _split_to_video_img_paths(
    file_paths,
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
    video_paths, img_paths = [], []

    for path in file_paths:
        if is_in_format(path, video_formats):
            video_paths.append(path)
        elif is_in_format(path, img_formats):
            img_paths.append(path)
        else:
            raise FormatNotSupportedError(
                "The format of path is not supported ({})".format(path)
            )
    return video_paths, img_paths


class FormatNotSupportedError(Exception):
    pass


def _create_chunks(file_paths, chunksize):
    if chunksize == 0:
        return file_paths
    else:
        chunk_starts = range(0, len(file_paths), chunksize)
        return [file_paths[i : i + chunksize] for i in chunk_starts]


def save_detections(
    detections, infile, overwrite=CONFIG["DETECT"]["YOLO"]["OVERWRITE"]
): 
    filepath = os.path.dirname(infile) + "/" + os.path.splitext(os.path.basename(infile))[0] + CONFIG["FILETYPES"]["DETECT"]
    exists = os.path.isfile(filepath)
    if overwrite or not exists:
        infile_path = Path(infile)
        outfile = str(infile_path.with_suffix(CONFIG["FILETYPES"]["DETECT"]))
        with open(outfile, "w") as f:
            json.dump(detections, f, indent=4)
        if exists:
            print("Detections file (" + os.path.basename(filepath) + ") overwritten") 
        else:
            print("Detections as " + os.path.basename(filepath) + " saved")
    else:
        print(os.path.basename(infile)+" already exists, was not overwritten")


# TODO: detect to df or gdf (geopandas)
def detect_df(
    files,
    filetypes: list = CONFIG["FILETYPES"]["VID"],
    model=None,
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
    ot_labels_enabled: bool = False,
):

    results = main(
        files=files,
        filetypes=filetypes,
        model=model,
        weights=weights,
        conf=conf,
        iou=iou,
        size=size,
        chunksize=chunksize,
        normalized=normalized,
        ot_labels_enabled=True,
    )

    for dets in results:
        # where dets is a list dets respective to files [dets_file_1, ..., dets_file_N]

        # Datenformat:
        # Geopandas?
        # | ix:filename | ix:frame | ix:detectionid | ...
        # ... shapely.geometry.box(xmin,ymin,xmax,ymax) | class | conf |

        # | ix:trackid | ix:filename | ix:frame | ix:detectionid | ...
        # ... shapely.geometry.box(xmin,ymin,xmax,ymax) | class | conf |

        # df["class"][""]
        # df.loc[123,"video.mp4", 543, 4), "class"]
        pass
