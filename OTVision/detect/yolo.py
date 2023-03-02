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

import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Union

import numpy
import torch
from cv2 import CAP_PROP_FPS, VideoCapture

from OTVision import dataformat, version
from OTVision.config import (
    CHUNK_SIZE,
    CONF,
    CONFIG,
    DETECT,
    FILETYPES,
    HALF_PRECISION,
    IMG_SIZE,
    IOU,
    NORMALIZED,
    VID,
    WEIGHTS,
    YOLO,
)
from OTVision.dataformat import (
    CLASS,
    CONFIDENCE,
    DATA,
    DETECTION,
    DETECTIONS,
    FILENAME,
    FILETYPE,
    HEIGHT,
    METADATA,
    MODEL,
    NUMBER_OF_FRAMES,
    OTDET_VERSION,
    OTVISION_VERSION,
    RECORDED_FPS,
    VIDEO,
    WIDTH,
    H,
    W,
    X,
    Y,
)
from OTVision.helpers.files import has_filetype
from OTVision.helpers.log import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)


class VideoFiletypeNotSupportedError(Exception):
    pass


class VideoFoundError(Exception):
    pass


class YOLOv5ModelNotFoundError(Exception):
    pass


def detect_video(
    file: Path,
    model: Union[torch.nn.Module, None] = None,
    weights: str = CONFIG[DETECT][YOLO][WEIGHTS],
    conf: float = CONFIG[DETECT][YOLO][CONF],
    iou: float = CONFIG[DETECT][YOLO][IOU],
    size: int = CONFIG[DETECT][YOLO][IMG_SIZE],
    half_precision: bool = CONFIG[DETECT][HALF_PRECISION],
    chunksize: int = CONFIG[DETECT][YOLO][CHUNK_SIZE],
    normalized: bool = CONFIG[DETECT][YOLO][NORMALIZED],
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
        raise VideoFiletypeNotSupportedError(
            (
                f"Filetype of '{file}' is not supported!",
                f"Only videos of filetype: '{CONFIG[FILETYPES][VID]}'",
            )
        )

    cap = VideoCapture(str(file))
    batch_no = 0

    got_frame = True
    t_loop_overhead = 0.0
    while got_frame:
        t_start = perf_counter()
        got_frame, img_batch = _get_batch_of_frames(cap, chunksize)

        t_get_batch = perf_counter()

        if not img_batch:
            break

        rgb_transformed_batch = [convert_bgr_to_rgb(frame) for frame in img_batch]

        t_trans = perf_counter()

        results = model(rgb_transformed_batch, size)

        t_det = perf_counter()

        _add_detection_results(yolo_detections, results, normalized)

        t_list = perf_counter()

        _log_batch_performances_stats(
            batch_no,
            t_start,
            t_get_batch,
            t_trans,
            t_det,
            t_list,
            t_loop_overhead,
            len(img_batch),
        )
        batch_no += 1

        width = cap.get(3)  # float
        height = cap.get(4)  # float
        fps = cap.get(CAP_PROP_FPS)  # float
        frames = cap.get(7)  # float
        t_loop_overhead = perf_counter() - t_list

    t2 = perf_counter()
    duration = t2 - t1
    det_fps = len(yolo_detections) / duration
    _log_overall_performance_stats(duration, det_fps)

    class_names = results.names

    det_config = _get_det_config(
        weights, conf, iou, size, half_precision, class_names, chunksize, normalized
    )
    vid_config = _get_vidconfig(file, width, height, fps, frames)
    return _convert_detections(yolo_detections, class_names, vid_config, det_config)


def convert_bgr_to_rgb(img_frame: numpy.ndarray) -> numpy.ndarray:
    return img_frame[:, :, ::-1]


def _get_batch_of_frames(
    video_capture: VideoCapture, batch_size: int
) -> tuple[bool, list[numpy.ndarray]]:
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
    t_get_batch: float,
    t_trans: float,
    t_det: float,
    t_list: float,
    t_loop_overhead: float,
    batch_size: int,
) -> None:
    batch_no_str = f"batch_no: {batch_no:d}"
    batch = f"batch: {t_get_batch - t_start:0.4f}"
    transformed_batch = f"trans: {t_trans - t_get_batch:0.4f}"
    det = f"det: {t_det - t_trans:0.4f}"
    add_list = f"list: {t_list - t_det:0.4f}"
    loop_overhead = f"loop_overhead: {t_loop_overhead:0.4f}"
    batch_len = f"batch_size: {batch_size:d}"
    fps = f"fps: {batch_size / (t_list - t_start):0.1f}"
    log_msg = (
        f"{batch_no_str}, {batch}, {transformed_batch}, {det}, "
        f"{add_list}, {loop_overhead}, {batch_len}, {fps}"
    )
    log.debug(log_msg)


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
def loadmodel(
    weights: str,
    conf: float,
    iou: float,
    force_reload: bool = False,
    half_precision: bool = False,
) -> Any:
    """Loads a local custom trained YOLOv5 model or a pretrained YOLOv5 model from torch
    hub.

    Args:
        weights (str): Path to custom model weights
        or model name i.e. 'yolov5s', 'yolov5m'.
        conf (float): The confidence threshold.
        iou (float): The IOU threshold.
        force_reload (bool, optional): Whether to force reload torch hub cache.
        Defaults to False.
        half_precision (bool, optional): Whether to use half precision (FP 16) to speed
        up inference. Only works for gpu. Defaults to False.

    Raises:
        ValueError: If the path to the model weights is not a .pt file.

    Returns:
        Any: The YOLOv5 model.
    """
    log.info(f"Try loading model {weights}")
    t1 = perf_counter()
    is_custom = Path(weights).is_file()

    try:
        if is_custom:
            model = _load_custom_model(weights=Path(weights), force_reload=force_reload)
        else:
            model = _load_pretrained_model(
                model_name=weights, force_reload=force_reload
            )
    except ValueError:
        raise
    except YOLOv5ModelNotFoundError:
        raise
    except Exception as e:
        if force_reload:
            # cache already force reloaded
            raise
        log.exception(e)
        log.info("Force reload cache and try again.")
        if is_custom:
            model = _load_custom_model(weights=Path(weights), force_reload=True)
        else:
            model = _load_pretrained_model(model_name=weights, force_reload=True)

    model.conf = conf
    model.iou = iou

    t2 = perf_counter()
    log.info(f"Model loaded in {round(t2 - t1)} sec")

    return model.half() if torch.cuda.is_available() and half_precision else model


def _load_pretrained_model(model_name: str, force_reload: bool) -> Any:
    """Load pretrained YOLOv5 model from torch hub.

    Args:
        model_name (str): As in ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
        force_reload (bool): Whether to force reload the cache.

    Raises:
        YOLOv5ModelNotFoundError: If YOLOv5 model could not be found on torch hub.
        ValueError: If the path to custom the model weights is not a .pt file.

    Returns:
        Any: The YOLOv5 model.
    """
    try:
        model = torch.hub.load(
            repo_or_dir="ultralytics/yolov5",
            model=model_name,
            pretrained=True,
            force_reload=force_reload,
        )
    except RuntimeError as re:
        if str(re).startswith("Cannot find callable"):
            raise YOLOv5ModelNotFoundError(
                f"YOLOv5 model: {model_name} does not found!"
            ) from re
        else:
            raise
    return model.cuda() if torch.cuda.is_available() else model.cpu()


def _load_custom_model(weights: Path, force_reload: bool) -> Any:
    """Load custom trained YOLOv5 model.

    Args:
        weights (Path): Path to model weights.
        force_reload (bool): Whether to force reload the cache.

    Raises:
        ValueError: If the path to the model weights is not a .pt file.

    Returns:
        Any: The YOLOv5 torch model.
    """
    if weights.suffix != ".pt":
        raise ValueError(f"Weights at '{weights}' is not a pt file!")

    model = torch.hub.load(
        repo_or_dir="ultralytics/yolov5",
        model="custom",
        path=weights,
        force_reload=force_reload,
    )
    return model.cuda() if torch.cuda.is_available() else model.cpu()


def _get_vidconfig(
    file: Path, width: int, height: int, fps: float, frames: int
) -> dict[str, Union[str, int, float]]:
    return {
        FILENAME: str(Path(file).stem),
        FILETYPE: str(Path(file).suffix),
        WIDTH: width,
        HEIGHT: height,
        RECORDED_FPS: fps,
        NUMBER_OF_FRAMES: frames,
    }


def _get_det_config(
    weights: Union[str, Path],
    conf: float,
    iou: float,
    image_size: int,
    half_precision: bool,
    classes: list[str],
    chunksize: int,
    normalized: bool,
) -> dict[str, Union[str, int, float, dict]]:
    return {
        OTVISION_VERSION: version.otvision_version(),
        MODEL: {
            dataformat.NAME: "YOLOv5",
            dataformat.WEIGHTS: str(weights),
            dataformat.IOU_THRESHOLD: iou,
            dataformat.IMAGE_SIZE: image_size,
            dataformat.MAX_CONFIDENCE: conf,
            dataformat.HALF_PRECISION: half_precision,
            dataformat.CLASSES: classes,
        },
        dataformat.CHUNKSIZE: chunksize,
        dataformat.NORMALIZED_BBOX: normalized,
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
                CLASS: names[int(yolo_bbox[5])],
                CONFIDENCE: yolo_bbox[4],
                X: yolo_bbox[0],
                Y: yolo_bbox[1],
                W: yolo_bbox[2],
                H: yolo_bbox[3],
            }

            detection.append(bbox)
        data = {str(no + 1): {DETECTIONS: detection}}
        # ?: Should every image have a det_config dict? Even if it is always the same?
        result.append({METADATA: {DETECTION: det_config}, DATA: data})
    return result


# TODO: Type hint nested list/dict during refactoring
def _convert_detections(
    yolo_detections: list,
    names: dict,
    vid_config: dict[str, Union[str, int, float]],
    det_config: dict[str, Union[str, int, float, dict]],
) -> dict[str, dict]:  # TODO: Type hint nested dict during refactoring
    data = {}
    for no, yolo_detection in enumerate(yolo_detections):
        # TODO: #81 Detections: Nested dict instead of dict of lists of dicts
        detection: list = []
        for yolo_bbox in yolo_detection:
            bbox = {
                CLASS: names[int(yolo_bbox[5])],
                CONFIDENCE: yolo_bbox[4],
                X: yolo_bbox[0],
                Y: yolo_bbox[1],
                W: yolo_bbox[2],
                H: yolo_bbox[3],
            }
            detection.append(bbox)
        data[str(no + 1)] = {DETECTIONS: detection}
    return {
        METADATA: {
            OTDET_VERSION: version.otdet_version(),
            VIDEO: vid_config,
            DETECTION: det_config,
        },
        DATA: data,
    }


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
