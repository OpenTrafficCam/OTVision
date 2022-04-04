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

import os
import shutil
from pathlib import Path

import progressbar

from OTVision.config import CONFIG
from OTVision.detect import detect


def _pngfiles(file):
    file = Path(file)
    dir = file.with_suffix("")
    pngfiles = dir.glob("obj_train_data/*.png")

    return pngfiles


def _fileList(file, suffix):
    file = Path(file)
    dir = file.with_suffix("")
    files = dir.glob("*.{}".format(suffix))
    return [str(file) for file in files]


def _unzip(file):
    file = Path(file)
    dir = file.with_suffix("")
    shutil.unpack_archive(file, dir)
    pngfiles = _pngfiles(file)
    files = [str(file) for file in pngfiles]
    return files


def _zip(file, pngs=False):
    file = Path(file)
    dir = file.with_suffix("")
    if not pngs:
        pngfiles = _pngfiles(file)
        for pngfile in pngfiles:
            pngfile.unlink()
    newfile = dir.parent / (file.stem + "_annotated")
    shutil.make_archive(newfile, "zip", root_dir=dir)
    shutil.rmtree(dir)


def _writenames(file, names):
    file = Path(file)
    dir = file.with_suffix("")
    objnames = dir / "obj.names"
    with open(objnames, "w") as f:
        for name in names:
            f.write((name + "\n"))


def _writebbox(file: str, xywhn: list):
    pngfiles = _pngfiles(file)

    for png in pngfiles:
        txt = png.with_suffix(".txt")
        detections = xywhn.pop(0)
        with open(txt, "a") as f:
            for detection in detections:
                x, y, w, h, conf, cls = detection
                line = "{cls:0.0f} {x:0.6f} {y:0.6f} {w:0.6f} {h:0.6f}".format(
                    x=x, y=y, w=w, h=h, cls=cls
                )
                f.write((line + "\n"))


def _writecvatlabels(file, results):
    # TODO: write cvat label data to file
    pass


def _pre_annotation(file, chunk_size):
    files = _unzip(file)
    xywhn, names = detect.main(
        files,
        filetypes=CONFIG["FILETYPES"]["IMG"],
        weights="yolov5x",
        chunksize=chunk_size,
        normalized=True,
        ot_labels_enabled=True,
    )
    _writebbox(file, xywhn)
    _writenames(file, names)
    _zip(file, pngs=False)


def check_isfile(file, chunk_size):
    if os.path.isfile(file):
        _pre_annotation(file, chunk_size)
    elif os.path.isdir(file):
        zipFiles = _fileList(file, "zip")
        for file in progressbar.progressbar(zipFiles):
            _pre_annotation(file, chunk_size)
