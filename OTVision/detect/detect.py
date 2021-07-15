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
from config import CONFIG
from helpers.files import get_files
from detect import yolo


# def main(paths, filetypes, det_config={}):
#     files = get_files(paths, filetypes)
#     multiple_videos(files, **det_config)


def main(
    files,
    # should default be CONFIG["FILETYPES"]["VID"]?
    #  allowing only one filetype or multiple? e.g. different video formats
    # TODO: Specify what filetypes are allowed, e.g. img and videos
    # filetype: str = ".mp4",
    filetypes: list = CONFIG["FILETYPES"]["VID"],
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
):  # sourcery skip: merge-dict-assign

    # if type(files) is not list:
    #     files = [files]

    model = yolo.loadmodel(weights, conf, iou)

    file_paths = get_files(paths=files, filetypes=filetypes)

    # split file paths to two groups -> videos | images
    # only when accepting multiple filetypes
    video_paths, frame_paths = _extract_video_paths(file_paths)

    frame_chunks = _create_chunks(frame_paths, chunksize)

    for path in video_paths:
        detections_videos = yolo.detect_video(
            file_path=path,
            model=model,
            weights=weights,
            conf=conf,
            iou=iou,
            size=size,
            chunksize=chunksize,
            normalized=normalized,
        )
        save_detections(detections_videos, path)

    detections_chunks = yolo.detect_chunks(
        file_chunks=frame_chunks,
        model=model,
        weights=weights,
        conf=conf,
        iou=iou,
        size=size,
        chunksize=chunksize,
        normalized=normalized,
    )

    # save detection information to corresponding frame path
    for frame_path, detection in zip(frame_paths, detections_chunks):
        save_detections(detection, frame_path)


def _extract_video_paths(file_paths):
    video_paths, other_paths = [], []

    for path in file_paths:
        if yolo.is_video(path):
            video_paths.append(path)
        else:
            other_paths.append(path)
    return video_paths, other_paths


def _create_chunks(file_paths, chunksize):
    if chunksize == 0:
        return file_paths
    else:
        chunk_starts = range(0, len(file_paths), chunksize)
        return [file_paths[i : i + chunksize] for i in chunk_starts]


def save_detections(
    detections, infile, overwrite=CONFIG["DETECT"]["YOLO"]["OVERWRITE"]
):
    if overwrite or not get_files(infile, CONFIG["FILETYPES"]["DETECT"]):
        infile_path = Path(infile)
        outfile = str(infile_path.with_suffix(CONFIG["FILETYPES"]["DETECT"]))
        with open(outfile, "w") as f:
            json.dump(detections, f, indent=4)
        if overwrite:
            print("Detections file overwritten")
        else:
            print("Detections file saved")
    else:
        print("Detections file already exists, was not overwritten")


if __name__ == "__main__":
    det_config = {"weights": "yolov5x", "conf": 0.25, "iou": 0.45, "size": 640}
