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

# TODO: docstrings in yolo

import torch
from time import perf_counter


def detect(
    files,
    weights: str = "yolov5s",
    conf: float = 0.25,
    iou: float = 0.45,
    size: int = 640,
    chunk_size: int = 0,
):

    if torch.cuda.is_available():
        model = torch.hub.load("ultralytics/yolov5", weights, pretrained=True).cuda()
    else:
        model = torch.hub.load("ultralytics/yolov5", weights, pretrained=True).cpu()

    model.conf = conf
    model.iou = iou

    print("Model loaded in {0:0.2f} s".format(perf_counter()))

    if chunk_size == 0:
        file_chunks = [files]
    else:
        chunk_starts = range(0, len(files), chunk_size)
        file_chunks = [files[i : i + chunk_size] for i in chunk_starts]

    start = perf_counter()
    xywhn = []

    for file_chunk in file_chunks:
        results = model(file_chunk, size=size)
        xywhn.extend([i.tolist() for i in results.xywhn])

    duration = perf_counter() - start
    fps = len(files) / duration
    print("All Chunks done in {0:0.2f} s ({1:0.2f} fps)".format(duration, fps))

    names = results.names

    # 'imgs'
    # 'pred'
    # 'names'
    # 'xyxy'
    # 'xywh'
    # 'xyxyn'
    # 'xywhn'
    # 'n'

    # 'display'
    # 'print'
    # 'show'
    # 'save'
    # 'render'
    # 'tolist'

    return xywhn, names


if __name__ == "__main__":
    files = [
        "OTVision/detect/frame_001000.PNG",
        "OTVision/detect/frame_001020.PNG",
    ]
    weights = "yolov5s"
    conf = 0.50
    iou = 0.45
    size = 640
    detect(files)
