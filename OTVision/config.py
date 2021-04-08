# OTVision: Python module to read and write configuration dict

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

from pathlib import Path
import json


config = {}

# FILETYPES
config["FILETYPES"]["VID"] = [
    ".mov",
    ".avi",
    ".mp4",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".wmv",
    ".mkv",
]
config["FILETYPES"]["IMG"] = [".jpg", ".jpeg", ".png"]

# LAST PATHS
config["LAST PATHS"]["FOLDER"] = None
config["LAST PATHS"]["VIDEO"] = None
config["LAST PATHS"]["DETECTIONS"] = None
config["LAST PATHS"]["TRACKS"] = None
config["LAST PATHS"]["CALIBRATION"] = None
config["LAST PATHS"]["REFPTS"] = None

# CONVERT
config["CONVERT"]["OUTPUT_FILETYPE"] = ".avi"
config["CONVERT"]["FPS"] = 25.0
config["CONVERT"]["OVERWRITE"] = True

# DETECT
config["DETECT"]["YOLO"]["WEIGHTS"] = "yolov5s"
config["DETECT"]["YOLO"]["CONF"] = 0.25
config["DETECT"]["YOLO"]["IOU"] = 0.45
config["DETECT"]["YOLO"]["SIZE"] = 640
config["DETECT"]["YOLO"]["CHUNKSIZE"] = 0
config["DETECT"]["YOLO"]["NORMALIZED"] = False
config["DETECT"]["YOLO"]["OVERWRTIE"] = True

# UNDISTORT
config["UNDISTORT"]["OVERWRTIE"] = False

# TRANSFORM
config["TRANSFORM"]["OVERWRTIE"] = False

# GUI
config["GUI"]["FONT"] = "Open Sans"
config["GUI"]["FONTSIZE"] = 12
config["GUI"]["WINDOW"]["LOCATION_X"] = 0
config["GUI"]["WINDOW"]["LOCATION_Y"] = 0


# TODO: #72 Overwrite default config with user config from user.conf (json file)
def _read(config_name: str = "user"):
    config_path = get_path(config_name=config_name)
    if config_path.suffix == ".otconf":
        if not config_path.is_file():
            print(f"{config_name}.otconf doesnt exist, load default.otconf instead")
            config_path = get_path(config_name="default")
            if not config_path.is_file():
                raise FileNotFoundError()
        with open(str(config_path)) as f:
            config = json.load(f)
        return config
    else:
        raise ValueError("Filetype for configuratuin has to be .otconf")


def _write(config: dict, config_name: str = "user"):
    config_path = get_path(config_name=config_name)
    if config_name == "default":
        answer = input("Sure you wanna overwrite default.otconf? [y/n]")
        if answer != "y":
            print("Configuration not saved, default.otconf not overwritten")
            return None
        print("default.otconf overwritten")
    with open(str(config_path), "w") as f:
        json.dump(config, f, indent=4)


def get_path(config_name="default"):
    config_path = Path(__file__).parents[0] / f"{config_name}.otconf"
    return config_path


if __name__ == "__main__":
    # config_dict = _read()
    # print(f"Config dict: {config_dict}")
    # _write(config_dict, config_name="user")
