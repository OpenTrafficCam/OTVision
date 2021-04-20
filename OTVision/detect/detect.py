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
    filetype: str = ".mp4",
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["SIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
):  # sourcery skip: merge-dict-assign

    # if type(files) is not list:
    #     files = [files]

    files = get_files(paths=files, filetypes=CONFIG["FILETYPES"]["VID"])

    model = yolo.loadmodel(weights, conf, iou)

    for file in files:
        detections = yolo.detect(
            file=file,
            model=model,
            size=size,
            chunksize=chunksize,
            normalized=normalized,
        )
        _save_detections(detections, file)


def _save_detections(detections, file):
    file = Path(file)
    filename = file.with_suffix(".otdet")
    with open(filename, "w") as f:
        json.dump(detections, f, indent=4)


if __name__ == "__main__":
    det_config = {"weights": "yolov5x", "conf": 0.25, "iou": 0.45, "size": 640}
