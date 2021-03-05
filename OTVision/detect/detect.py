# Copyright (C) 2021 OpenTrafficCam Contributors
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


def main(paths, filetype, det_config={}):
    files = get_files(paths, filetype)
    multiple_videos(files, **det_config)


def multiple_videos(
    files,
    weights: str = "yolov5x",
    conf: float = 0.25,
    iou: float = 0.45,
    size: int = 640,
):

    det_config = {"weights": weights, "conf": conf, "iou": iou, "size": size}

    if type(files) is not list:
        files = [files]

    model = yolo.loadmodel(weights, conf, iou)

    for file in files:

        yolo_detections, names = yolo.detect(
            files=file,
            model=model,
            size=size,
            chunk_size=0,
            normalized=True,
        )

        detections = yolo.convert_detections(yolo_detections, names, det_config)
        _save_detections(detections, file)


def _save_detections(detections, file):
    file = Path(file)
    filename = file.with_suffix(".otdet")
    with open(filename, "w") as f:
        json.dump(detections, f, indent=4)


if __name__ == "__main__":
    det_config = {"weights": "yolov5x", "conf": 0.25, "iou": 0.45, "size": 640}
