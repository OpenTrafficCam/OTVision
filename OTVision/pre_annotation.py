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

import shutil
from pathlib import Path
from time import perf_counter
from typing import Union

import progressbar

from OTVision.detect import detect
from OTVision.helpers.files import get_files, unzip
from OTVision.helpers.log import log


def _zip_annotated_dir(cvat_yolo_dir: Path, img_type: str, pngs: bool = False) -> Path:
    to_be_zipped = cvat_yolo_dir
    if not pngs:
        img_paths = get_files(paths=[to_be_zipped], filetypes=[img_type])
        for img_path in img_paths:
            img = Path(img_path)
            img.unlink()
    new_file = to_be_zipped.parent / f"{to_be_zipped.name}_annotated"
    shutil.make_archive(base_name=str(new_file), format="zip", root_dir=to_be_zipped)
    shutil.rmtree(path=to_be_zipped)
    return new_file.with_name(f"{new_file.stem}.zip")


def _write_class_labels(cvat_yolo_dir: Path, class_labels: dict) -> None:
    obj_names = cvat_yolo_dir / "obj.names"
    with open(obj_names, "w") as f:
        for name in class_labels.values():
            f.write((name + "\n"))


def _write_bbox(
    cvat_yolo_dir: Path,
    img_type: str,
    bboxes_xywhn: list,
    class_labels: dict,
    filter_classes: Union[dict, None],
) -> None:
    image_paths = get_files([cvat_yolo_dir], filetypes=[img_type])
    assert len(image_paths) == len(bboxes_xywhn)

    for img_path, detections in zip(image_paths, bboxes_xywhn):
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
) -> Path:
    cvat_yolo_dir = unzip(cvat_yolo_zip)

    detect_results = detect.main(
        paths=[cvat_yolo_dir],
        filetypes=[img_type],
        weights=model_weights,
        chunksize=chunk_size,
        normalized=True,
        ot_labels_enabled=True,
    )
    # Ensure that no unbound variables are passed to _write_bbox
    bboxes_in_xywhn_format: list = []
    class_labels: dict = {}

    if detect_results:
        bboxes_in_xywhn_format, class_labels = detect_results

    _write_bbox(
        cvat_yolo_dir, img_type, bboxes_in_xywhn_format, class_labels, filter_classes
    )

    if filter_classes is not None:
        class_labels = filter_classes

    _write_class_labels(cvat_yolo_dir, class_labels)
    return _zip_annotated_dir(cvat_yolo_dir, img_type, pngs=False)


def main(
    path: Path,
    model_weights: str,
    chunk_size: int = 1000,
    filter_classes: Union[None, dict] = None,
    img_type: str = "png",
) -> None:
    """_summary_
    TODO: Write docstrings
    Args:
        path (Path): _description_
        model_weights (str): _description_
        chunk_size (int, optional): _description_. Defaults to 1000.
        filter_classes (Union[None, dict], optional): _description_. Defaults to None.
        img_type (str, optional): _description_. Defaults to "png".

    Raises:
        OSError: _description_
        OSError: _description_

    ### Example

    ```python
    from OTVision import pre_annotation

    file_path = "/path/to/zip/file.zip"

    model_weights = "yolov5s.pt"
    chunk_size = 200
    classes = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "bus",
        5: "truck",
    }

    pre_annotation.main(file_path, model_weights, chunk_size, classes)
    ```
    """
    log.info("Starting")

    if not path.exists():
        raise OSError(f"Path at: '{path}' does not exist!")

    if path.is_file():
        _pre_annotate(path, model_weights, chunk_size, filter_classes, img_type)
    elif path.is_dir():
        zip_files = get_files([path], ["zip"])
        for f in progressbar.progressbar(zip_files):
            _pre_annotate(f, model_weights, chunk_size, filter_classes, img_type)
    else:
        raise OSError(f"Path at: '{path}' is neither a file nor a directory!")

    log.info("Done in {0:0.2f} s".format(perf_counter()))
