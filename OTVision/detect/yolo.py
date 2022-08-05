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
from xml.dom.minidom import Attr

import torch
from cv2 import CAP_PROP_FPS, VideoCapture

from OTVision.config import CONFIG
from OTVision.helpers.files import is_in_format


class NoVideoError(Exception):
    pass


class VideoFoundError(Exception):
    pass


def detect_video(
    file_path,
    model=None,
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
):
    """Detect and classify bounding boxes in videos using YOLOv5

    Args:
        files (str ot list of str): files to detect.
        model (yolo object): Yolo model to detect with.
        weights (str, optional): Weigths, if no model passed. Defaults to "yolov5s".
        conf (float, optional): Output confidence, if no model passed. Defaults to 0.25.
        iou (float, optional): IOU param, if no model passed. Defaults to 0.45.
        size (int, optional): Frame size for detection. Defaults to 640.
        chunksize (int, optional): Number of files per detection chunk. Defaults to 0.
        normalized (bool, optional): Coords in % of image/frame size (True) or pixels
        (False). Defaults to False.

    Returns:
        [type]: [description]
    """
    if model is None:
        model = loadmodel(weights, conf, iou)

    yolo_detections = []
    t1 = perf_counter()

    if not is_in_format(file_path, CONFIG["FILETYPES"]["VID"]):
        raise NoVideoError("The file: {} is not a video!".format(file_path))

    cap = VideoCapture(file_path)
    batch_no = 0

    print("Run detection on video: {}".format(file_path))

    got_frame = True
    while got_frame:
        got_frame, img_batch = _get_batch_of_frames(cap, chunksize)

        if not img_batch:
            break

        t_start = perf_counter()

        # What purpose does this transformation have
        transformed_batch = list(map(lambda frame: frame[:, :, ::-1], img_batch))

        t_trans = perf_counter()

        results = model(transformed_batch, size)

        t_det = perf_counter()

        _add_detection_results(yolo_detections, results, normalized)

        t_list = perf_counter()

        _print_batch_performances_stats(
            batch_no, t_start, t_trans, t_det, t_list, len(img_batch)
        )
        batch_no += 1

        width = cap.get(3)  # float
        height = cap.get(4)  # float
        fps = cap.get(CAP_PROP_FPS)  # float
        frames = cap.get(7)  # float

    t2 = perf_counter()
    duration = t2 - t1
    det_fps = len(yolo_detections) / duration
    _print_overall_performance_stats(duration, det_fps)

    class_names = results.names

    det_config = _get_det_config(weights, conf, iou, size, chunksize, normalized)
    vid_config = _get_vidconfig(file_path, width, height, fps, frames)
    detections = _convert_detections(
        yolo_detections, class_names, vid_config, det_config
    )

    return detections


def detect_images(
    file_chunks,
    model=None,
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
    ot_labels_enabled: bool = False,
):
    """Detect and classify bounding boxes in images/frames using YOLOv5

    Args:
        files (str ot list of str): files to detect.
        model (yolo object): Yolo model to detect with.
        weights (str, optional): Weigths, if no model passed. Defaults to "yolov5s".
        conf (float, optional): Output confidence, if no model passed. Defaults to 0.25.
        iou (float, optional): IOU param, if no model passed. Defaults to 0.45.
        size (int, optional): Frame size for detection. Defaults to 640.
        chunksize (int, optional): Number of files per detection chunk. Defaults to 0.
        normalized (bool, optional): Coords in % of image/frame size (True) or pixels
        (False). Defaults to False.
        ot_labels_enabled (bool, optional): returns [detections, names] where detections
        consist of bounding boxes but without any annotations and the class name index
        (True) or returns the detections in otdet format(False). Defaults to False.

    Returns:
        [type]: [description]
    """
    yolo_detections = []

    if not file_chunks:
        return [], [] if ot_labels_enabled else yolo_detections

    if model is None:
        model = loadmodel(weights, conf, iou)

    t1 = perf_counter()

    if _containsvideo(file_chunks):
        raise VideoFoundError(
            "List of paths given to detect_chunks function shouldn't contain any videos"
        )
    img_batch = 0
    print("Run detection on images")
    for chunk in file_chunks:
        t_start = perf_counter()
        results = model(chunk, size=size)
        t_det = perf_counter()
        _add_detection_results(yolo_detections, results, normalized)
        img_batch += 1
        print(
            (
                f"img_batch_no: {img_batch:d}, "
                f"det:{t_det - t_start:0.4f}, "
                f"batch_size: {len(chunk):d}, "
                f"fps: {chunksize / (t_det - t_start):0.1f}"
            )
        )

    t2 = perf_counter()
    duration = t2 - t1
    det_fps = len(yolo_detections) / duration
    _print_overall_performance_stats(duration, det_fps)

    names = results.names
    if ot_labels_enabled:
        return [yolo_detections, names]
    else:
        det_config = _get_det_config(weights, conf, iou, size, chunksize, normalized)
        detections = _convert_detections_chunks(yolo_detections, names, det_config)
        return detections


def _get_batch_of_frames(video_capture, batch_size):
    """Reads the the next batch_size frames from VideoCapture.

    Args:
        video_capture (obj): VideoCapture instance.
        batch_size (int): batch size.

    Returns:
        gotFrame (bool): True if there are more frames to read.
        False if no more frames can be read.
        batch(list): batch of frames.
    """
    batch = []
    gotFrame = False
    for _ in range(0, batch_size):
        gotFrame, img = video_capture.read()
        if gotFrame:
            batch.append(img)
        else:
            break
    return gotFrame, batch


def _print_overall_performance_stats(duration, det_fps):
    print("All Chunks done in {0:0.2f} s ({1:0.2f} fps)".format(duration, det_fps))


def _print_batch_performances_stats(
    batch_no, t_start, t_trans, t_det, t_list, batch_size
):
    batch_no = "batch_no: {:d}".format(batch_no)
    transformed_batch = "trans: {:0.4f}".format(t_trans - t_start)
    det = "det: {:0.4f}".format(t_det - t_start)
    add_list = "list: {:0.4f}".format(t_list - t_det)
    batch_len = "batch_size: {:d}".format(batch_size)
    fps = "fps: {:0.1f}".format(batch_size / (t_det - t_start))

    print(
        "{0}, {1}, {2}, {3}, {4}, {5}".format(
            batch_no, transformed_batch, det, add_list, batch_len, fps
        )
    )


def _add_detection_results(detections, results, normalized):
    """Adds detection result to an existing list.

    Args:
        detections (list): the existing list containing detections.
        results (list): detection results.
        normalized (bool): True if results are normalized. False otherwise.

    Returns:
        list: the detections list with the newly added
    """
    if normalized:
        detections.extend([i.tolist() for i in results.xywhn])
    else:
        detections.extend([i.tolist() for i in results.xywh])


def loadmodel(weights, conf=0.25, iou=0.25):
    HUB_WEIGHTS = [
        "yolov5n",
        "yolov5n6",
        "yolov5s",
        "yolov5s6",
        "yolov5m",
        "yolov5m6",
        "yolov5l",
        "yolov5l6",
        "yolov5x",
        "yolov5x6",
    ]

    t1 = perf_counter()

    if Path(weights).is_file():
        model = torch.hub.load(
            repo_or_dir="ultralytics/yolov5",  # cv516Buaa/tph-yolov5 ?
            model="custom",
            path=weights,
            # source="local",
            force_reload=True,
        )
        # model.amp = False ?
        # model = torch.jit.load(weights) ?
    elif weights in torch.hub.list(github="ultralytics/yolov5", force_reload=True):

        print(torch.hub.list(github="ultralytics/yolov5", force_reload=True))

        if torch.cuda.is_available():
            model = torch.hub.load(
                repo_or_dir="ultralytics/yolov5",
                model=weights,
                pretrained=True,
                force_reload=True,
            ).cuda()
        else:
            model = torch.hub.load(
                repo_or_dir="ultralytics/yolov5",
                model=weights,
                pretrained=True,
                force_reload=True,
            ).cpu()
    else:
        raise AttributeError(
            "weights has to be path to .pt or valid model name from https://pytorch.org/hub/ultralytics_yolov5/"
        )

    model.conf = conf
    model.iou = iou

    t2 = perf_counter()
    print("Model loaded in {0:0.2f} s".format(t2 - t1))
    return model


def _get_vidconfig(file, width, height, fps, frames):
    return {
        "file": str(Path(file).stem),
        "filetype": str(Path(file).suffix),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
    }


def _get_det_config(weights, conf, iou, size, chunksize, normalized):
    return {
        "detector": "YOLOv5",
        "weights": weights,
        "conf": conf,
        "iou": iou,
        "size": size,
        "chunksize": chunksize,
        "normalized": normalized,
    }


def _convert_detections_chunks(yolo_detections, names, det_config):
    result = []
    for no, yolo_detection in enumerate(yolo_detections):
        data = {}
        detection = []
        for yolo_bbox in yolo_detection:
            bbox = {
                "class": names[int(yolo_bbox[5])],
                "conf": yolo_bbox[4],
                "x": yolo_bbox[0],
                "y": yolo_bbox[1],
                "w": yolo_bbox[2],
                "h": yolo_bbox[3],
            }
            detection.append(bbox)
        data[str(no + 1)] = {"classified": detection}
        result.append({"det_config": det_config, "data": data})

    return result


def _convert_detections(yolo_detections, names, vid_config, det_config):
    data = {}
    for no, yolo_detection in enumerate(yolo_detections):
        # TODO: #81 Detections: Nested dict instead of dict of lists of dicts
        detection = []
        for yolo_bbox in yolo_detection:
            bbox = {
                "class": names[int(yolo_bbox[5])],
                "conf": yolo_bbox[4],
                "x": yolo_bbox[0],
                "y": yolo_bbox[1],
                "w": yolo_bbox[2],
                "h": yolo_bbox[3],
            }
            detection.append(bbox)
        data[str(no + 1)] = {"classified": detection}
    return {"vid_config": vid_config, "det_config": det_config, "data": data}


def _createchunks(chunksize, files):
    # TODO: Remove method
    if type(files) is str:
        return [files]
    elif _containsvideo(files):
        return files
    elif chunksize == 0:
        return [files]
    else:
        chunk_starts = range(0, len(files), chunksize)
        return [files[i : i + chunksize] for i in chunk_starts]


def _containsvideo(file_chunks):
    if len(file_chunks) == 0:
        return False

    if type(file_chunks[0]) is str:
        file = Path(file_chunks[0])
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
        if file.suffix in vid_formats:
            return True
    return False
