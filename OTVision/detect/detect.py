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

from helpers.files import get_files

from detect import yolo


def main(paths, filetypes, det_config={}):
    files = get_files(paths, filetypes)
    multiple_videos(files, **det_config)


def multiple_videos(
    files,
    weights: str = "yolov5x",
    conf: float = 0.25,
    iou: float = 0.45,
    size: int = 640,
    chunksize: int = 0,
    normalized: bool = False,
):

    print("normalized")
    print(normalized)
    det_config = {
        "weights": weights,
        "conf": conf,
        "iou": iou,
        "size": size,
        "chunksize": chunksize,
        "normalized": normalized,
    }

    if type(files) is not list:
        files = [files]

    model = yolo.loadmodel(weights, conf, iou)

    for file in files:

        yolo_detections, names, width, height, fps, frames = yolo.detect(
            files=file,
            model=model,
            size=size,
            chunk_size=chunksize,
            normalized=normalized,
        )

        vid_config = {}
        vid_config["file"] = str(Path(file).stem)
        vid_config["filetype"] = str(Path(file).suffix)
        vid_config["width"] = width
        vid_config["height"] = height
        vid_config["fps"] = fps
        vid_config["frames"] = frames

        detections = yolo.convert_detections(
            yolo_detections, names, vid_config, det_config
        )

        print(detections)
        _save_detections(detections, file)


def _save_detections(detections, file):
    file = Path(file)
    filename = file.with_suffix(".otdet")
    with open(filename, "w") as f:
        json.dump(detections, f, indent=4)


if __name__ == "__main__":
    det_config = {"weights": "yolov5x", "conf": 0.25, "iou": 0.45, "size": 640}
