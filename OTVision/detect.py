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

from OTVision.detect import detect

if __name__ == "__main__":
    test_path = Path(__file__).parents[1] / "tests" / "data"
    test_path = str(test_path)
    det_config = {
        "weights": "yolov5s",
        "conf": 0.25,
        "iou": 0.45,
        "size": 640,
        "chunksize": 5,
        "normalized": False,
        "ot_labels_enabled": True
    }
    detect.main(test_path, [".mp4", ".jpeg", ".jpg"], **det_config)
