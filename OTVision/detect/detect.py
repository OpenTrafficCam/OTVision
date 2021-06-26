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


import torch
from time import perf_counter
from cv2 import VideoCapture, CAP_PROP_FPS
from config import CONFIG

import json

from pathlib import Path
from config import CONFIG
from helpers.files import get_files
from detect import yolo


# def main(paths, filetypes, det_config={}):
#     files = get_files(paths, filetypes)
#     multiple_videos(files, **det_config)

class Detection:
    def __init__(
        self,
        filetype: str = ".mp4",
        weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
        conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
        iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
        size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
        chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
        normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
    ):
        self.model = yolo.loadmodel(weights, conf, iou)
        self.filetype = filetype
        self.weights = weights
        self.conf = conf
        self.iou = iou
        self.size = size
        self.chunksize = chunksize
        self.normalized = normalized

    def main(self, files):
        filePaths = get_files(paths=files, filetypes=CONFIG["FILETYPES"]["VID"])

        for videoPath in filePaths:
            detections = self.detect(videoPath)
            save_detections(detections, videoPath)

    def detect(self, pathToVideo):
        yolo_detections = []
        t1 = perf_counter()

        if _isVideo(pathToVideo):
            cap = VideoCapture(pathToVideo)
            batch_no = 0
            #TODO while gotframe
            while True:
                gotFrame, imgBatch = self._getBatchOfFrames(cap)
                t_start = perf_counter()

                if not gotFrame:
                    print("Gotframe: {}".format(gotFrame))
                    print("Should check for images in batch")

                # What purpose does this transformation have
                transformedBatch = list(map(lambda frame: frame[:, :, ::-1], imgBatch))

                if len(imgBatch) == 0:
                    break

                t_trans = perf_counter()

                results = self.model(transformedBatch, self.size)

                t_det = perf_counter()

                if self.normalized:
                    yolo_detections.extend([i.tolist() for i in results.xywhn])
                else:
                    yolo_detections.extend([i.tolist() for i in results.xywh])
                t_list = perf_counter()
                t_batch = perf_counter()

                self._printBatchPerformanceStats(batch_no, t_start, t_trans, t_det,
                                                 t_list, t_batch)
                batch_no += 1

                if not gotFrame:
                    break

            width = cap.get(3)  # float
            height = cap.get(4)  # float
            fps = cap.get(CAP_PROP_FPS)  # float
            frames = cap.get(7)  # floa
        # TODO: inference file chunks that are not in video format

        t2 = perf_counter()
        duration = t2 - t1
        det_fps = len(yolo_detections) / duration
        self._printOverallPerformanceStats(duration, det_fps)

        names = results.names

        if _isVideo(pathToVideo):
            # TODO: accessing private methods! 
            det_config = yolo._get_det_config(self.weights,
                                              self.conf, self.iou, self.size,
                                              self.chunksize, self.normalized)
            vid_config = yolo._get_vidconfig(pathToVideo, width, height, fps, frames)
            detections = yolo._convert_detections(yolo_detections, names, vid_config, 
                                                  det_config)
        else:
            detections = [yolo_detections, names]

        return detections

    def _printOverallPerformanceStats(self, duration, det_fps):
        print("All Chunks done in {0:0.2f} s ({1:0.2f} fps)".format(duration, det_fps))

    def _printBatchPerformanceStats(self, batch_no, t_start, t_trans, t_det, t_list, t_batch):
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

    def _getBatchOfFrames(self, cap):
        batch = []
        for frame in range(0, self.chunksize):
            gotFrame, img = cap.read()
            if gotFrame:
                batch.append(img)
            else:
                break
        return gotFrame, batch


def _isVideo(pathToVideo):
    video_formats = CONFIG["FILETYPES"]["VID"]
    videoFile = Path(pathToVideo)

    for format in video_formats:
        if videoFile.suffix in video_formats:
            return True
        else:
            return False


def main(
    files,
    filetype: str = ".mp4",
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
):  # sourcery skip: merge-dict-assign

    # if type(files) is not list:
    #     files = [files]

    model = yolo.loadmodel(weights, conf, iou)

    """ for file in files:
        detections = yolo.detect(
            file=file,
            model=model,
            size=size,
            chunksize=chunksize,
            normalized=normalized,
    ) """

    detections = yolo.detect(
        file=files,
        model=model,
        size=size,
        chunksize=chunksize,
        normalized=normalized,
    )

    for file in files:
        save_detections(detections, file)

    """ detectFilesOptimized(files, model, weights, conf, iou, size, chunksize, normalized) """


def save_detections(
    detections, infile, overwrite=CONFIG["DETECT"]["YOLO"]["OVERWRITE"]
):
    if overwrite or not get_files(infile, CONFIG["FILETYPES"]["DETECT"]):
        infile_path = Path(infile)
        outfile = str(infile_path.with_suffix(CONFIG["FILETYPES"]["DETECT"]))
        with open(outfile, "w") as f:
            json.dump(detections, f, indent=4)
        if overwrite:
            print("Detections file overwritten")
        else:
            print("Detections file saved")
    else:
        print("Detections file already exists, was not overwritten")


if __name__ == "__main__":
    det_config = {"weights": "yolov5x", "conf": 0.25, "iou": 0.45, "size": 640}
