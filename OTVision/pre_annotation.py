"""
OTVision module to pre-annotate images using detect.py
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

# TODO: docstrings in pre_annotation

import os
import shutil
from pathlib import Path
from time import perf_counter
from typing import Union

import progressbar

from OTVision.detect import detect
from OTVision.helpers.files import get_files, unzip
from OTVision.helpers.log import log


def _zip_annotated_dir(cvat_yolo_dir: Union[str, Path], img_type: str, pngs=False):
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


def _write_class_labels(cvat_yolo_dir: Union[str, Path], class_labels: list[str]):
    cvat_yolo_dir = Path(cvat_yolo_dir)
    obj_names = cvat_yolo_dir / "obj.names"
    with open(obj_names, "w") as f:
        for name in class_labels.values():
            f.write((name + "\n"))


def _write_bbox(
    cvat_yolo_dir: str,
    img_type: str,
    xywhn: list,
    class_labels: dict,
    filter_classes: dict,
):
    image_paths = get_files(cvat_yolo_dir, filetypes=img_type)
    assert len(image_paths) == len(xywhn)

    for img_path, detections in zip(image_paths, xywhn):
        annotation_txt = Path(img_path).with_suffix(".txt")

        with open(annotation_txt, "w") as f:
            for detection in detections:
                x, y, w, h, _, _cls = detection  # [x, y, w, h, conf, class]
                if filter_classes is not None:
                    cls_name = class_labels[_cls]
                    if cls_name in list(filter_classes.values()):
                        new_cls = list(filter_classes.keys())[
                            list(filter_classes.values()).index(cls_name)
                        ]
                        if new_cls in filter_classes.keys():
                            line = (
                                "{cls:0.0f} {x:0.6f} {y:0.6f} {w:0.6f} {h:0.6f}".format(
                                    x=x, y=y, w=w, h=h, cls=new_cls
                                )
                            )
                            f.write((line + "\n"))
                else:
                    line = "{cls:0.0f} {x:0.6f} {y:0.6f} {w:0.6f} {h:0.6f}".format(
                        x=x, y=y, w=w, h=h, cls=_cls
                    )
                    f.write((line + "\n"))


def _pre_annotate(
    cvat_yolo_zip: Path,
    model_weights: str,
    chunk_size: int,
    filter_classes: Union[None, dict],
    img_type: str,
):
    yolo_cvat_dir = unzip(cvat_yolo_zip)
    bboxes_in_xywhn_format, class_labels = detect.main(
        paths=yolo_cvat_dir,
        filetypes=img_type,
        weights=model_weights,
        chunksize=chunk_size,
        normalized=True,
        ot_labels_enabled=True,
    )

    _write_bbox(
        yolo_cvat_dir, img_type, bboxes_in_xywhn_format, class_labels, filter_classes
    )

    if filter_classes is not None:
        class_labels = filter_classes

    _write_class_labels(yolo_cvat_dir, class_labels)
    return _zip_annotated_dir(yolo_cvat_dir, img_type, pngs=False)


def main(
    file: Union[str, Path],
    model_weights: str,
    chunk_size: int = 1000,
    filter_classes: Union[None, dict] = None,
    img_type: str = "png",
):
    log.info("Starting")
    if os.path.isfile(file):
        _pre_annotate(file, model_weights, chunk_size, filter_classes, img_type)
    elif os.path.isdir(file):
        zip_files = get_files(file, "zip")
        for file in progressbar.progressbar(zip_files):
            _pre_annotate(file, model_weights, chunk_size, filter_classes, img_type)
    log.info("Done in {0:0.2f} s".format(perf_counter()))


if __name__ == "__main__":
    file_path = (
        "/Users/michaelheilig/Downloads/task_800x600_cloudy_h5m_aov60deg_"
        + "intersection_priority_mondercangeintersection5-2022_08_19_10_23_31"
        + "-yolo 1.1.zip"
    )
    model_weights = "yolov5s.pt"
    chunk_size = 200
    filter_classes = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "bus",
        5: "truck",
    }

    main(file_path, model_weights, chunk_size, filter_classes)
