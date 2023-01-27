"""
OTVision module to detect objects using yolov5
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

from pathlib import Path
from time import perf_counter
from typing import Any, Union

import torch
from cv2 import CAP_PROP_FPS, VideoCapture

from OTVision.config import CONFIG
from OTVision.helpers.files import has_filetype
from OTVision.helpers.log import log


class NoVideoError(Exception):
    pass


class VideoFoundError(Exception):
    pass


def detect_video(
    file: Path,
    model: Union[torch.nn.Module, None] = None,
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
) -> dict[str, dict]:  # TODO: Type hint nested dict during refactoring
    """Detect and classify bounding boxes in videos using YOLOv5

    Args:
        file (Path): files to detect.
        model (torch.nn.Module): Yolo model to detect with.
        weights (str, optional): Weigths, if no model passed. Defaults to "yolov5s".
        conf (float, optional): Output confidence, if no model passed. Defaults to 0.25.
        iou (float, optional): IOU param, if no model passed. Defaults to 0.45.
        size (int, optional): Frame size for detection. Defaults to 640.
        chunksize (int, optional): Number of files per detection chunk. Defaults to 0.
        normalized (bool, optional): Coords in % of image/frame size (True) or pixels
            (False). Defaults to False.

    Returns:
        dict[str, dict]: Dict with subdicts of metadata and actual detections
    """
    if model is None:
        model = loadmodel(weights, conf, iou)

    yolo_detections: list = []
    t1 = perf_counter()

    if not has_filetype(file, CONFIG["FILETYPES"]["VID"]):
        raise NoVideoError(f"The file: {file} is not a video!")

    cap = VideoCapture(str(file))
    batch_no = 0

    log.info(f"Run detection on video: {file}")

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

        _log_batch_performances_stats(
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
    _log_overall_performance_stats(duration, det_fps)

    class_names = results.names

    det_config = _get_det_config(weights, conf, iou, size, chunksize, normalized)
    vid_config = _get_vidconfig(file, width, height, fps, frames)
    return _convert_detections(yolo_detections, class_names, vid_config, det_config)


def detect_images(
    file_chunks: list[list[Path]],
    model: Union[torch.nn.Module, None] = None,
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
    ot_labels_enabled: bool = False,
) -> Union[tuple[list, dict], list]:
    """Detect and classify bounding boxes in images/frames using YOLOv5

    Args:
        file_chunks (list of list of Path): files to detect.
        model (torch.nn.Module, optional): Yolo model to detect with.
        weights (str, optional): Weigths, if no model passed. Defaults to "yolov5s".
        conf (float, optional): Output confidence, if no model passed. Defaults to 0.25.
        iou (float, optional): IOU param, if no model passed. Defaults to 0.45.
        size (int, optional): Frame size for detection. Defaults to 640.
        chunksize (int, optional): Number of files per detection chunk. Defaults to 0.
        normalized (bool, optional): Coords in % of image/frame size (True) or pixels
        (False). Defaults to False.
        ot_labels_enabled (bool, optional): returns [detections, names] where detections
        consist of bounding boxes but without any annotations and the class name index
        (True) or returns the dœetections in otdet format(False). Defaults to False.

    Returns:
        [type]: [description]
    """
    yolo_detections: list = []
    if not file_chunks:
        return ([], {}) if ot_labels_enabled else yolo_detections
    if model is None:
        model = loadmodel(weights, conf, iou)
    t1 = perf_counter()
    if _containsvideo(file_chunks):
        raise VideoFoundError(
            "List of paths given to detect_chunks function shouldn't contain any videos"
        )

    log.info("Run detection on images")
    for img_batch, chunk in enumerate(file_chunks, start=1):
        t_start = perf_counter()
        results = model(chunk, size=size)
        t_det = perf_counter()
        _add_detection_results(yolo_detections, results, normalized)
        str_batch_no = f"img_batch_no: {img_batch:d}"
        str_det_time = f"det:{t_det - t_start:0.4f}"
        str_batch_size = f"batch_size: {len(chunk):d}"
        str_fps = f"fps: {chunksize / (t_det - t_start):0.1f}"
        log.info(f"{str_batch_no}, {str_det_time}, {str_batch_size}, {str_fps}")

    t2 = perf_counter()
    duration = t2 - t1
    det_fps = len(yolo_detections) / duration
    _log_overall_performance_stats(duration, det_fps)
    class_labels: dict = results.names
    if ot_labels_enabled:
        return (yolo_detections, class_labels)
    det_config = _get_det_config(weights, conf, iou, size, chunksize, normalized)
    return _convert_detections_chunks(yolo_detections, class_labels, det_config)


def _get_batch_of_frames(
    video_capture: VideoCapture, batch_size: int
) -> tuple[bool, list]:
    """Reads the the next batch_size frames from VideoCapture.

    Args:
        video_capture (cv2.VideoCapture): VideoCapture instance.
        batch_size (int): batch size.

    Returns:
        gotFrame (bool): True if there are more frames to read.
        False if no more frames can be read.
        batch(list): batch of frames.
    """
    batch = []
    gotFrame: bool = False
    for _ in range(batch_size):
        gotFrame, img = video_capture.read()
        if gotFrame:
            batch.append(img)
        else:
            break
    return gotFrame, batch


def _log_overall_performance_stats(duration: float, det_fps: float) -> None:
    log.info("All Chunks done in {0:0.2f} s ({1:0.2f} fps)".format(duration, det_fps))


def _log_batch_performances_stats(
    batch_no: int,
    t_start: float,
    t_trans: float,
    t_det: float,
    t_list: float,
    batch_size: int,
) -> None:
    batch_no_str = "batch_no: {:d}".format(batch_no)
    transformed_batch = "trans: {:0.4f}".format(t_trans - t_start)
    det = "det: {:0.4f}".format(t_det - t_start)
    add_list = "list: {:0.4f}".format(t_list - t_det)
    batch_len = "batch_size: {:d}".format(batch_size)
    fps = "fps: {:0.1f}".format(batch_size / (t_det - t_start))
    log_msg = (
        f"{batch_no_str}, {transformed_batch}, {det}, {add_list}, {batch_len}, {fps}"
    )
    log.info(log_msg)  # BUG: #162 Logs twice from yolo.py (with and without formatting)


def _add_detection_results(
    detections: list,  # TODO: Type hint nested list/dict during refactoring
    results: Any,  # ?: Type hint from YOLOv5 repo from yolov5.models.common.Detections
    normalized: bool,
) -> None:
    """Adds detection result to the list of detections provided.

    Args:
        detections (list): the existing list containing detections.
        results (Any): detection results.
        normalized (bool): True if results are normalized. False otherwise.
    """
    if normalized:
        detections.extend([i.tolist() for i in results.xywhn])
    else:
        detections.extend([i.tolist() for i in results.xywh])


# TODO: loadmodel: Arg "local_weights" [Path](optional) that overrides "weights" [str]
def loadmodel(weights: Union[Path, str], conf: float, iou: float) -> Any:
    """Loads model from torch.hub using custom local weights or standard weights from
        torch.hub

    Args:
        weights (Union[Path, str]): yolov5 weights file or default model name
        conf (float): Confidence threshold for detection
        iou (float): IOU threshold for detection

    Raises:
        AttributeError: If weights is neither .pt-file nor valid model name

    Returns:
        Any: yolov5 model
    """

    log.info(f"Try loading model {weights}")
    t1 = perf_counter()

    if Path(weights).is_file() and Path(weights).suffix == ".pt":
        model = torch.hub.load(
            repo_or_dir="ultralytics/yolov5",
            model="custom",
            path=weights,
            force_reload=True,
        )
    elif weights in torch.hub.list(github="ultralytics/yolov5", force_reload=True):

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
            "weights has to be path to .pt or valid model name "
            "from https://pytorch.org/hub/ultralytics_yolov5/"
        )

    model.conf = conf
    model.iou = iou

    t2 = perf_counter()
    log.info(f"Model loaded in {round(t2 - t1)} sec")
    return model


def _get_vidconfig(
    file: Path, width: int, height: int, fps: float, frames: int
) -> dict[str, Union[str, int, float]]:
    return {
        "file": str(Path(file).stem),
        "filetype": str(Path(file).suffix),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
    }


def _get_det_config(
    weights: Union[str, Path],
    conf: float,
    iou: float,
    size: int,
    chunksize: int,
    normalized: bool,
) -> dict[str, Union[str, int, float]]:
    return {
        "detector": "YOLOv5",
        "weights": str(weights),
        "conf": conf,
        "iou": iou,
        "size": size,
        "chunksize": chunksize,
        "normalized": normalized,
    }


# TODO: Type hint nested list/dict during refactoring
def _convert_detections_chunks(
    yolo_detections: list, names: dict, det_config: dict[str, Union[str, int, float]]
) -> list:
    result = []
    for no, yolo_detection in enumerate(yolo_detections):
        detection: list = []
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
        data = {str(no + 1): {"classified": detection}}
        # ?: Should every image have a det_config dict? Even if it is always the same?
        result.append({"metadata": {"det": det_config}, "data": data})
    return result


# TODO: Type hint nested list/dict during refactoring
def _convert_detections(
    yolo_detections: list,
    names: dict,
    vid_config: dict[str, Union[str, int, float]],
    det_config: dict[str, Union[str, int, float]],
) -> dict[str, dict]:  # TODO: Type hint nested dict during refactoring
    data = {}
    for no, yolo_detection in enumerate(yolo_detections):
        # TODO: #81 Detections: Nested dict instead of dict of lists of dicts
        detection: list = []
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
    return {"metadata": {"vid": vid_config, "det": det_config}, "data": data}


def _containsvideo(file_chunks: list[list[Path]]) -> bool:
    if not file_chunks:
        return False

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

    for file_chunk in file_chunks:
        for file in file_chunk:
            if file.suffix in vid_formats:
                return True

    return False
