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

from pathlib import Path
from time import perf_counter
import os
import json

import torch
from cv2 import VideoCapture


def detect(
    files,
    weights: str = "yolov5s",
    conf: float = 0.25,
    iou: float = 0.45,
    size: int = 640,
    chunk_size: int = 0,
    resulttype="xywhn",
):
    """[summary]

    Args:
        files ([type]): Detect and classify bounding boxes in images/frames with YoloV5
        weights (str, optional): [description]. Defaults to "yolov5s".
        conf (float, optional): [description]. Defaults to 0.25.
        iou (float, optional): [description]. Defaults to 0.45.
        size (int, optional): [description]. Defaults to 640.
        chunk_size (int, optional): [description]. Defaults to 0.
        resulttype (str, optional): Reference for returend bbox ('xyxy', 'xywh',
        'xyxyn', 'xywhn'. "n" stands for normalized coordinates of x and y in percent).
        Defaults to "xywhn".

    Returns:
        [type]: [description]
    """

    model = _loadmodel(weights, conf, iou)

    file_chunks = _createchunks(chunk_size, files)

    bboxes = []
    t1 = perf_counter()
    if _containsvideo(file_chunks):
        for file_chunk in file_chunks:
            cap = VideoCapture(file_chunk)
            gotframe, img = cap.read()
            frame_no = 0
            while gotframe:
                t_start = perf_counter()
                img = img[:, :, ::-1]
                t_trans = perf_counter()
                results = model(img, size=size)
                t_det = perf_counter()
                if resulttype == "xywhn":
                    bboxes.extend([i.tolist() for i in results.xywhn])
                elif resulttype == "xyxy":
                    bboxes.extend([i.tolist() for i in results.xyxy])
                t_list = perf_counter()
                gotframe, img = cap.read()
                t_frame = perf_counter()
                print(
                    "frame_no: {0:0.4f}, trans: {1:0.4f}, det: {2:0.4f}, list: {3:0.4f}, frame: {4:0.4f}, fps:{5:0.1f}".format(
                        frame_no,
                        t_trans - t_start,
                        t_det - t_start,
                        t_list - t_det,
                        t_frame - t_list,
                        1 / (t_frame - t_start),
                    )
                )
                frame_no += 1

    else:
        for file_chunk in file_chunks:
            results = model(file_chunk, size=size)
            if resulttype == "xywhn":
                bboxes.extend([i.tolist() for i in results.xywhn])
            elif resulttype == "xyxy":
                bboxes.extend([i.tolist() for i in results.xyxy])

    t2 = perf_counter()
    duration = t2 - t1
    fps = len(bboxes) / duration
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

    return bboxes, names


def _loadmodel(weights, conf, iou):
    t1 = perf_counter()

    if torch.cuda.is_available():
        model = torch.hub.load("ultralytics/yolov5", weights, pretrained=True).cuda()
    else:
        model = torch.hub.load("ultralytics/yolov5", weights, pretrained=True).cpu()

    model.conf = conf
    model.iou = iou

    t2 = perf_counter()
    print("Model loaded in {0:0.2f} s".format(t2 - t1))
    return model


def _createchunks(chunk_size, files):
    if type(files) is str:
        file_chunks = [files]
    elif _containsvideo(files):
        file_chunks = files
    elif chunk_size == 0:
        file_chunks = [files]
    else:
        chunk_starts = range(0, len(files), chunk_size)
        file_chunks = [files[i : i + chunk_size] for i in chunk_starts]
    return file_chunks


def _containsvideo(file_chunks):
    vid_formats = [
        ".mov",
        ".avi",
        ".mp4",
        ".mpg",
        ".mpeg",
        ".m4v",
        ".wmv",
        ".mkv",
    ]
    if type(file_chunks[0]) is str:
        file = Path(file_chunks[0])
        if file.suffix in vid_formats:
            return True
    return False


def detect_df(
    files,
    weights: str = "yolov5x",
    conf: float = 0.25,
    iou: float = 0.45,
    size: int = 640,
):

    results = detect(files, weights, conf, iou, size)

    tensors = results.xywhn

    for tensor in tensors:
        tensor.tolist()

        # Datenformat:
        # Geopandas?
        # | ix:filename | ix:frame | ix:detectionid | shapely.geometry.box(xmin,ymin,xmax,ymax) | class | conf |
        # | ix:trackid | ix:filename | ix:frame | ix:detectionid | shapely.geometry.box(xmin,ymin,xmax,ymax) | class | conf |

        # df["class"][""]
        # df.loc[123,"video.mp4", 543, 4), "class"]


def save_bboxes(files, bboxes, names, style="json_iou"):
    detections_dict = {}
    for frame_no, frame in enumerate(bboxes):
        bbox_dict_list = []
        for bbox in frame:
            bbox_values_dict = {
                "label": names[int(bbox[5])],
                "confidence": bbox[4],
                "ymax": int(bbox[3]),
                "xmax": int(bbox[2]),
                "ymin": int(bbox[1]),
                "xmin": int(bbox[0]),
                "ymid": int(int(bbox[1]) + (int(bbox[3]) - int(bbox[1])) / 2),
                "xmid": int(int(bbox[0]) + (int(bbox[2]) - int(bbox[0])) / 2),
                "height": int(bbox[3]) - int(bbox[1]),
                "width": int(bbox[2]) - int(bbox[0]),
            }
            bbox_dict_list.append(bbox_values_dict)
        detections_dict[str(frame_no + 1)] = {"classified": bbox_dict_list}
    print(detections_dict)
    filename = os.path.splitext(files)[0] + "_detections.json"
    json.dump(detections_dict, open(filename, "w"))


if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent.absolute()
    video_1 = root_path / r"tests/data/testvideo_1.mkv"
    video_2 = root_path / r"tests/data/testvideo_2.mkv"
    # files = [video_1.__str__(), video_2.__str__()]
    files = video_1.__str__()

    weights = "yolov5x"
    conf = 0.50
    iou = 0.45
    size = 640
    chunk_size = 0
    bboxes, names = detect(
        files=files,
        resulttype="xyxy",
    )
    print(bboxes)
    save_bboxes(files, bboxes, names)
