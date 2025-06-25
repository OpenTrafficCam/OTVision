"""
OTVision config module for setting default values
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


import logging
from pathlib import Path

from OTVision.application.config import (
    AVAILABLE_WEIGHTS,
    CALIBRATIONS,
    COL_WIDTH,
    CONF,
    CONVERT,
    DEFAULT_FILETYPE,
    DELETE_INPUT,
    DETECT,
    DETECT_END,
    DETECT_START,
    DETECTIONS,
    EXPECTED_DURATION,
    FILETYPES,
    FONT,
    FONT_SIZE,
    FPS_FROM_FILENAME,
    FRAME_WIDTH,
    GUI,
    HALF_PRECISION,
    IMG,
    IMG_SIZE,
    INPUT_FPS,
    IOU,
    LAST_PATHS,
    LOCATION_X,
    LOCATION_Y,
    LOG,
    LOG_LEVEL_CONSOLE,
    LOG_LEVEL_FILE,
    NORMALIZED,
    OTC_ICON,
    OUTPUT_FILETYPE,
    OUTPUT_FPS,
    OVERWRITE,
    PATHS,
    REFPTS,
    ROTATION,
    RUN_CHAINED,
    SEARCH_SUBDIRS,
    SIGMA_H,
    SIGMA_IOU,
    SIGMA_L,
    T_MIN,
    T_MISS_MAX,
    TRACK,
    TRACKS,
    TRANSFORM,
    UNDISTORT,
    VID,
    VID_ROTATABLE,
    VIDEOS,
    WEIGHTS,
    WINDOW,
    WRITE_VIDEO,
    YOLO,
    Config,
)
from OTVision.application.config_parser import ConfigParser
from OTVision.helpers.log import LOGGER_NAME
from OTVision.plugin.yaml_serialization import YamlDeserializer

log = logging.getLogger(LOGGER_NAME)


def parse_user_config(yaml_file: Path | str) -> Config:
    """Parses a custom OTVision user config yaml file.

    Args:
        yaml_file (Path |str): The absolute Path to the config file.
    """
    user_config_file = Path(yaml_file)
    deserializer = YamlDeserializer()
    user_config = ConfigParser(deserializer).parse(user_config_file)
    CONFIG.update(user_config.to_dict())
    return user_config


# sourcery skip: merge-dict-assign
CONFIG: dict = {}

# LOGGING
CONFIG[LOG] = {}
CONFIG[LOG][LOG_LEVEL_CONSOLE] = "WARNING"
CONFIG[LOG][LOG_LEVEL_FILE] = "DEBUG"

# FOLDERS
CONFIG[SEARCH_SUBDIRS] = True

# FILETYPES
CONFIG[DEFAULT_FILETYPE] = {}
CONFIG[DEFAULT_FILETYPE][VID] = ".mp4"
CONFIG[DEFAULT_FILETYPE][IMG] = ".jpg"
CONFIG[DEFAULT_FILETYPE][DETECT] = ".otdet"
CONFIG[DEFAULT_FILETYPE][TRACK] = ".ottrk"
CONFIG[DEFAULT_FILETYPE][REFPTS] = ".otrfpts"
CONFIG[FILETYPES] = {}
CONFIG[FILETYPES][VID] = [
    ".avi",
    ".mkv",
    ".mov",
    ".mp4",
]
CONFIG[FILETYPES][VID_ROTATABLE] = [
    ".mov",
    ".mp4",
]
CONFIG[FILETYPES][IMG] = [".jpg", ".jpeg", ".png"]
CONFIG[FILETYPES][DETECT] = [".otdet"]
CONFIG[FILETYPES][TRACK] = [".ottrk"]
CONFIG[FILETYPES][REFPTS] = [".otrfpts"]
CONFIG[FILETYPES][TRANSFORM] = [".gpkg"]

# LAST PATHS
CONFIG[LAST_PATHS] = {}
CONFIG[LAST_PATHS][VIDEOS] = []
CONFIG[LAST_PATHS][DETECTIONS] = []
CONFIG[LAST_PATHS][TRACKS] = []
CONFIG[LAST_PATHS][CALIBRATIONS] = []
CONFIG[LAST_PATHS][REFPTS] = []

# CONVERT
CONFIG[CONVERT] = {}
CONFIG[CONVERT][PATHS] = []
CONFIG[CONVERT][RUN_CHAINED] = True
CONFIG[CONVERT][OUTPUT_FILETYPE] = ".mp4"
CONFIG[CONVERT][INPUT_FPS] = 20.0
CONFIG[CONVERT][OUTPUT_FPS] = 20.0
CONFIG[CONVERT][FPS_FROM_FILENAME] = True
CONFIG[CONVERT][DELETE_INPUT] = False
CONFIG[CONVERT][ROTATION] = 0
CONFIG[CONVERT][OVERWRITE] = True

# DETECT
CONFIG[DETECT] = {}
CONFIG[DETECT][PATHS] = []
CONFIG[DETECT][RUN_CHAINED] = True
CONFIG[DETECT][YOLO] = {}
CONFIG[DETECT][YOLO][WEIGHTS] = "yolov8s"
CONFIG[DETECT][YOLO][AVAILABLE_WEIGHTS] = [
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
]
CONFIG[DETECT][YOLO][CONF] = 0.25
CONFIG[DETECT][YOLO][IOU] = 0.45
CONFIG[DETECT][YOLO][IMG_SIZE] = 640
CONFIG[DETECT][YOLO][NORMALIZED] = False
CONFIG[DETECT][EXPECTED_DURATION] = None
CONFIG[DETECT][OVERWRITE] = True
CONFIG[DETECT][HALF_PRECISION] = False
CONFIG[DETECT][DETECT_START] = None
CONFIG[DETECT][DETECT_END] = None
CONFIG[DETECT][WRITE_VIDEO] = False

# TRACK
CONFIG[TRACK] = {}
CONFIG[TRACK][PATHS] = []
CONFIG[TRACK][RUN_CHAINED] = True
CONFIG[TRACK][IOU] = {}
CONFIG[TRACK][IOU][SIGMA_L] = 0.27  # 0.272
CONFIG[TRACK][IOU][SIGMA_H] = 0.42  # 0.420
CONFIG[TRACK][IOU][SIGMA_IOU] = 0.38  # 0.381
CONFIG[TRACK][IOU][T_MIN] = 5
CONFIG[TRACK][IOU][T_MISS_MAX] = 51  # 51
CONFIG[TRACK][OVERWRITE] = True

# UNDISTORT
CONFIG[UNDISTORT] = {}
CONFIG[UNDISTORT][OVERWRITE] = False

# TRANSFORM
CONFIG[TRANSFORM] = {}
CONFIG[TRANSFORM][PATHS] = []
CONFIG[TRANSFORM][RUN_CHAINED] = True
CONFIG[TRANSFORM][OVERWRITE] = True

# GUI
CONFIG[GUI] = {}
CONFIG[GUI][OTC_ICON] = str(
    Path(__file__).parents[0] / r"view" / r"helpers" / r"OTC.ico"
)
CONFIG[GUI][FONT] = "Open Sans"
CONFIG[GUI][FONT_SIZE] = 12
CONFIG[GUI][WINDOW] = {}
CONFIG[GUI][WINDOW][LOCATION_X] = 0
CONFIG[GUI][WINDOW][LOCATION_Y] = 0
CONFIG[GUI][FRAME_WIDTH] = 80
CONFIG[GUI][COL_WIDTH] = 50
PAD = {"padx": 10, "pady": 10}

# TODO: #72 Overwrite default config with user config from user.conf (json file)
