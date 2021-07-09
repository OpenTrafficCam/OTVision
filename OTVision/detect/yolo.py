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
import torch
from cv2 import VideoCapture, CAP_PROP_FPS
from config import CONFIG


def detect(
    file,
    model=None,
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
):  # sourcery skip: inline-immediately-returned-variable
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

    Returns:
        [type]: [description]
    """
    if model is None:
        model = loadmodel(weights, conf, iou)

    file_chunks = _createchunks(chunksize, file)

    yolo_detections = []
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
                if normalized:
                    yolo_detections.extend([i.tolist() for i in results.xywhn])
                else:
                    yolo_detections.extend([i.tolist() for i in results.xywh])
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
            width = cap.get(3)  # float
            height = cap.get(4)  # float
            fps = cap.get(CAP_PROP_FPS)  # float
            frames = cap.get(7)  # float

    else:
        for file_chunk in file_chunks:
            results = model(file_chunk, size=size)
            if normalized:
                yolo_detections.extend([i.tolist() for i in results.xywhn])
            else:
                yolo_detections.extend([i.tolist() for i in results.xywh])

    t2 = perf_counter()
    duration = t2 - t1
    det_fps = len(yolo_detections) / duration
    print("All Chunks done in {0:0.2f} s ({1:0.2f} fps)".format(duration, det_fps))

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

    # TODO: #74 Refactor this and detect into a main()
    if _containsvideo(file_chunks):
        det_config = _get_det_config(weights, conf, iou, size, chunksize, normalized)
        vid_config = _get_vidconfig(file, width, height, fps, frames)
        detections = _convert_detections(yolo_detections, names, vid_config, det_config)
    else:
        detections = [yolo_detections, names]

    return detections


class NoVideoException(Exception):
    pass


class VideoFoundException(Exception):
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
    if model is None:
        model = loadmodel(weights, conf, iou)

    yolo_detections = []
    t1 = perf_counter()

    if not is_video(file_path):
        raise NoVideoException("The file: {} is not a video!".format(file_path))

    cap = VideoCapture(file_path)
    batch_no = 0
    # TODO while gotframe
    # what happens when gotframe false -> while loop never executed when while got_frame:
    # we are working in batches
    while True:
        got_frame, img_batch = _get_batch_of_frames(cap, chunksize)
        t_start = perf_counter()

        # What purpose does this transformation have
        transformedBatch = list(map(lambda frame: frame[:, :, ::-1], img_batch))

        if not img_batch:
            break

        t_trans = perf_counter()

        results = model(transformedBatch, size)

        t_det = perf_counter()

        _add_detection_results(yolo_detections, results, normalized)

        t_list = perf_counter()
        t_batch = perf_counter()

        _print_batch_performances_stats(
            batch_no, t_start, t_trans, t_det, t_list, t_batch
        )
        batch_no += 1

        width = cap.get(3)  # float
        height = cap.get(4)  # float
        fps = cap.get(CAP_PROP_FPS)  # float
        frames = cap.get(7)  # float

        if not got_frame:
            break

    t2 = perf_counter()
    duration = t2 - t1
    det_fps = len(yolo_detections) / duration
    _print_overall_performance_stats(duration, det_fps)

    names = results.names

    det_config = _get_det_config(weights, conf, iou, size, chunksize, normalized)
    vid_config = _get_vidconfig(file_path, width, height, fps, frames)
    detections = _convert_detections(yolo_detections, names, vid_config, det_config)

    """     if is_video(path_to_file):
            det_config = _get_det_config(
                weights, conf, iou, size, chunksize, normalized
            )
            vid_config = _get_vidconfig(path_to_file, width, height, fps, frames)
            detections = _convert_detections(
                yolo_detections, names, vid_config, det_config
            )
        else:
            detections = [yolo_detections, names] """

    return detections


def detect_chunks(
    file_chunks,
    model=None,
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],  # TODO: not needed
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
):
    yolo_detections = []

    if not file_chunks:
        return yolo_detections

    if model is None:
        model = loadmodel(weights, conf, iou)

    t1 = perf_counter()

    if _containsvideo(file_chunks):
        raise VideoFoundException(
            "List of paths given to detect_chunks function shouldn't contain any videos"
        )
    for chunk in file_chunks:
        results = model(chunk, size=size)
        _add_detection_results(yolo_detections, results, normalized)

    t2 = perf_counter()
    duration = t2 - t1
    det_fps = len(yolo_detections) / duration
    _print_overall_performance_stats(duration, det_fps)

    names = results.names

    detections = [yolo_detections, names]
    return detections


# TODO: Where should this class be? Helper class?
def is_video(pathToVideo):
    video_formats = CONFIG["FILETYPES"]["VID"]
    videoFile = Path(pathToVideo)

    if videoFile.suffix in video_formats:
        return True
    else:
        return False


def _get_batch_of_frames(cap, batchSize):
    batch = []
    for frame in range(0, batchSize):
        gotFrame, img = cap.read()
        if gotFrame:
            batch.append(img)
        else:
            break
    return gotFrame, batch


def _print_overall_performance_stats(duration, det_fps):
    print("All Chunks done in {0:0.2f} s ({1:0.2f} fps)".format(duration, det_fps))


def _print_batch_performances_stats(batch_no, t_start, t_trans, t_det, t_list, t_batch):
    print(
        "batch_no: {0:0.4f}, trans: {1:0.4f}, det: {2:0.4f}, list: {3:0.4f}, batch: {4:0.4f}, fps:{5:0.1f}".format(
            batch_no,
            t_trans - t_start,
            t_det - t_start,
            t_list - t_det,
            t_batch - t_list,
            1 / (t_batch - t_start),
        )
    )


def _add_detection_results(detections, results, normalized):
    if normalized:
        detections.extend([i.tolist() for i in results.xywhn])
    else:
        detections.extend([i.tolist() for i in results.xywh])


def loadmodel(weights, conf, iou):
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


# TODO: detect to df or gdf (geopandas)
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


if __name__ == "__main__":
    test_path = Path(__file__).parents[2] / "tests" / "data"
    video_1 = str(test_path / "testvideo_1.mkv")
    video_2 = str(test_path / "testvideo_2.mkv")
    # files = [video_1, video_2]
    files = video_1

    weights = "yolov5x"
    conf = 0.50
    iou = 0.45
    size = 640
    chunksize = 0

    bboxes, names = detect(
        files=files,
        normalized="xyxy",
    )
    print(bboxes)
    main(files, bboxes, names)
