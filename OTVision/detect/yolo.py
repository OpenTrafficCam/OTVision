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


import torch


def detect(
    files,
    weights: str = "yolov5s",
    conf: float = 0.25,
    iou: float = 0.45,
    size: int = 640,
):

    model = torch.hub.load("ultralytics/yolov5", weights, pretrained=True)

    model.conf = conf
    model.iou = iou

    results = model(files, size=size)

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

    return results


if __name__ == "__main__":
    files = [
        r"D:\git\OpenTrafficCam\OTVision\obj_train_data\frame_001000.PNG",
        r"D:\git\OpenTrafficCam\OTVision\obj_train_data\frame_001020.PNG",
    ]
    weights = "yolov5s"
    conf = 0.50
    iou = 0.45
    size = 640
    detect(files)
