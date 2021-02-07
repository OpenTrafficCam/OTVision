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

from pathlib import Path
import shutil

from detect.yolo import detect


def _pngfiles(file):
    file = Path(file)
    dir = file.with_suffix("")
    pngfiles = dir.glob("obj_train_data/*.png")

    return pngfiles


def _unzip(file):
    file = Path(file)
    dir = file.with_suffix("")
    shutil.unpack_archive(file, dir)
    pngfiles = _pngfiles(file)
    files = [str(file) for file in pngfiles]
    return files


def _zip(file):
    file = Path(file)
    dir = file.with_suffix("")
    newfile = dir.parent / (file.stem + "_annotated")
    shutil.make_archive(newfile, "zip", root_dir=dir)
    shutil.rmtree(dir)


def _writenames(file, results):
    file = Path(file)
    dir = file.with_suffix("")
    objnames = dir / "obj.names"
    names = results.names
    with open(objnames, "w") as f:
        for name in names:
            f.write((name + "\n"))


def _writebbox(file, results):
    pngfiles = _pngfiles(file)

    itensor = 0
    for png in pngfiles:
        txt = png.with_suffix(".txt")
        detections = results.xywhn[itensor].tolist()
        for detection in detections:
            x, y, w, h, conf, cls = detection
            line = "{cls:0.0f} {x:0.6f} {y:0.6f} {w:0.6f} {h:0.6f}".format(
                x=x, y=y, w=w, h=h, cls=cls
            )
            with open(txt, "a") as f:
                f.write((line + "\n"))
        itensor += 1


def _writecvatlabels(file, results):
    # TODO: write cvat label data to file
    pass


def pre_annotation(file):
    files = _unzip(file)
    results = detect(files, weights="yolov5x")
    _writebbox(file, results)
    _writenames(file, results)
    _zip(file)


if __name__ == "__main__":
    file = r"E:\Downloads\task_quercam13_2019-03-26_08-30-00-2021_02_07_22_06_05-yolo 1.1.zip"
    pre_annotation(file)
