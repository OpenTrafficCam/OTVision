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


from pathlib import Path
from zipfile import ZipFile

from detect.yolo import detect


def _unzip(file):
    file = Path(file)
    yolozip = ZipFile(file)

    dir = file.with_suffix("")
    yolozip.extractall(dir)

    pngfiles = dir.glob("**/*.png")
    files = [str(file) for file in pngfiles]
    return files


def _writenames():
    pass


def _writebbox():
    pass


if __name__ == "__main__":
    file = r"E:\Downloads\task_quercam13_2019-03-26_08-30-00-2021_02_07_00_20_49-yolo 1.1.zip"
    files = _unzip(file)
    results = detect(files)
