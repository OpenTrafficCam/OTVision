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

# TODO: docstrings in pre_annotation

import shutil
import os
from time import perf_counter

from pathlib import Path
import progressbar

from OTVision.detect import detect
from OTVision.helpers.files import get_files
from OTVision.helpers.files import unzip


def _zip_annotated_dir(cvat_yolo_dir, img_type, pngs=False):
    to_be_zipped = Path(cvat_yolo_dir)
    if not pngs:
        img_paths = get_files(to_be_zipped, filetypes=img_type)
        for img_path in img_paths:
            img = Path(img_path)
            img.unlink()
    new_file = to_be_zipped.parent / f"{to_be_zipped.name}_annotated"
    shutil.make_archive(new_file, "zip", root_dir=to_be_zipped)
    shutil.rmtree(to_be_zipped)
    return new_file.with_name(f"{new_file.stem}.zip")


def _write_class_labels(cvat_yolo_dir, class_labels):
    cvat_yolo_dir = Path(cvat_yolo_dir)
    obj_names = cvat_yolo_dir / "obj.names"
    with open(obj_names, "w") as f:
        for name in class_labels:
            f.write((name + "\n"))


def _write_bbox(cvat_yolo_dir: str, img_type: str, xywhn: list):
    image_paths = get_files(cvat_yolo_dir, filetypes=img_type)
    assert len(image_paths) == len(xywhn)

    for img_path, detections in zip(image_paths, xywhn):
        annotation_txt = Path(img_path).with_suffix(".txt")

        with open(annotation_txt, "w") as f:
            for detection in detections:
                x, y, w, h, _, _cls = detection  # [x, y, w, h, conf, class]
                line = "{cls:0.0f} {x:0.6f} {y:0.6f} {w:0.6f} {h:0.6f}".format(
                    x=x, y=y, w=w, h=h, cls=_cls
                )
                f.write((line + "\n"))


def _pre_annotate(cvat_yolo_zip, model_weights, chunk_size, img_type):
    yolo_cvat_dir = unzip(cvat_yolo_zip)
    bboxes_in_xywhn_format, class_labels = detect.main(
        files=yolo_cvat_dir,
        filetypes=img_type,
        weights=model_weights,
        chunksize=chunk_size,
        normalized=True,
        ot_labels_enabled=True,
    )
    _write_bbox(yolo_cvat_dir, img_type, bboxes_in_xywhn_format)
    _write_class_labels(yolo_cvat_dir, class_labels)
    yolo_cvat_annotated = _zip_annotated_dir(yolo_cvat_dir, img_type, pngs=False)
    return yolo_cvat_annotated


def main(file, model_weights, chunk_size, img_type="png"):
    print("Starting")
    if os.path.isfile(file):
        _pre_annotate(file, model_weights, chunk_size, img_type)
    elif os.path.isdir(file):
        zip_files = get_files(file, "zip")
        for file in progressbar.progressbar(zip_files):
            _pre_annotate(file, model_weights, chunk_size, img_type)
    print("Done in {0:0.2f} s".format(perf_counter()))


if __name__ == "__main__":

    path = r"C:\Users\MichaelHeilig\Downloads\annotation_data\task_wolfartsweierer straße #9-2021_05_05_14_38_18-yolo 1.1.zip"
    chunk_size = 100
    main(path, "yolov5s", chunk_size)
