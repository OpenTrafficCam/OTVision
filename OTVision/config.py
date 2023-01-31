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


from pathlib import Path

# sourcery skip: merge-dict-assign
CONFIG: dict = {}


# FOLDERS
CONFIG["SEARCH_SUBDIRS"] = True

# FILETYPES
CONFIG["DEFAULT_FILETYPE"] = {}
CONFIG["DEFAULT_FILETYPE"]["VID"] = ".mp4"
CONFIG["DEFAULT_FILETYPE"]["IMG"] = ".jpg"
CONFIG["DEFAULT_FILETYPE"]["DETECT"] = ".otdet"
CONFIG["DEFAULT_FILETYPE"]["TRACK"] = ".ottrk"
CONFIG["DEFAULT_FILETYPE"]["REFPTS"] = ".otrfpts"
CONFIG["FILETYPES"] = {}
CONFIG["FILETYPES"]["VID"] = [
    ".avi",
    ".mkv",
    ".m4v",
    ".mov",
    ".mp4",
    ".mpg",
    ".mpeg",
    ".wmv",
]
CONFIG["FILETYPES"]["IMG"] = [".jpg", ".jpeg", ".png"]
CONFIG["FILETYPES"]["VID_IMG"] = CONFIG["FILETYPES"]["VID"] + CONFIG["FILETYPES"]["IMG"]
CONFIG["FILETYPES"]["DETECT"] = [".otdet"]
CONFIG["FILETYPES"]["TRACK"] = [".ottrk"]
CONFIG["FILETYPES"]["REFPTS"] = [".otrfpts"]
CONFIG["FILETYPES"]["TRANSFORM"] = [".gpkg"]

# LAST PATHS
CONFIG["LAST PATHS"] = {}
CONFIG["LAST PATHS"]["VIDEOS"] = []
CONFIG["LAST PATHS"]["DETECTIONS"] = []
CONFIG["LAST PATHS"]["TRACKS"] = []
CONFIG["LAST PATHS"]["CALIBRATIONS"] = []
CONFIG["LAST PATHS"]["REFPTS"] = []

# CONVERT
CONFIG["CONVERT"] = {}
CONFIG["CONVERT"]["RUN_CHAINED"] = True
CONFIG["CONVERT"][
    "FFMPEG_URL"
] = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
CONFIG["CONVERT"]["FFMPEG_PATH"] = str(
    Path(__file__).parents[0] / r"convert" / r"ffmpeg.exe"
)
CONFIG["CONVERT"]["OUTPUT_FILETYPE"] = ".mp4"
CONFIG["CONVERT"]["INPUT_FPS"] = 20.0
CONFIG["CONVERT"]["OUTPUT_FPS"] = 20.0
CONFIG["CONVERT"]["FPS_FROM_FILENAME"] = True
CONFIG["CONVERT"]["DELETE_INPUT"] = False
CONFIG["CONVERT"]["OVERWRITE"] = True
CONFIG["CONVERT"]["DEBUG"] = False

# DETECT
CONFIG["DETECT"] = {}
CONFIG["DETECT"]["RUN_CHAINED"] = True
CONFIG["DETECT"]["OTLABELS_ENABLES"] = False
CONFIG["DETECT"]["YOLO"] = {}
CONFIG["DETECT"]["YOLO"]["WEIGHTS"] = "yolov5s"
CONFIG["DETECT"]["YOLO"]["AVAILABLEWEIGHTS"] = [
    "yolov5s",
    "yolov5m",
    "yolov5l",
    "yolov5x",
]
CONFIG["DETECT"]["YOLO"]["CONF"] = 0.25
CONFIG["DETECT"]["YOLO"]["IOU"] = 0.45
CONFIG["DETECT"]["YOLO"]["IMGSIZE"] = 640
CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"] = 1
CONFIG["DETECT"]["YOLO"]["NORMALIZED"] = False
CONFIG["DETECT"]["OVERWRITE"] = True
CONFIG["DETECT"]["DEBUG"] = False
CONFIG["DETECT"]["HALF_PRECISION"] = False
CONFIG["DETECT"]["FORCE_RELOAD_TORCH_HUB_CACHE"] = False

# TRACK
CONFIG["TRACK"] = {}
CONFIG["TRACK"]["RUN_CHAINED"] = True
CONFIG["TRACK"]["IOU"] = {}
CONFIG["TRACK"]["IOU"]["SIGMA_L"] = 0.27  # 0.272
CONFIG["TRACK"]["IOU"]["SIGMA_H"] = 0.42  # 0.420
CONFIG["TRACK"]["IOU"]["SIGMA_IOU"] = 0.38  # 0.381
CONFIG["TRACK"]["IOU"]["T_MIN"] = 5
CONFIG["TRACK"]["IOU"]["T_MISS_MAX"] = 51  # 51
CONFIG["TRACK"]["OVERWRITE"] = True
CONFIG["TRACK"]["DEBUG"] = False

# UNDISTORT
CONFIG["UNDISTORT"] = {}
CONFIG["UNDISTORT"]["OVERWRTIE"] = False
CONFIG["UNDISTORT"]["DEBUG"] = False

# TRANSFORM
CONFIG["TRANSFORM"] = {}
CONFIG["TRANSFORM"]["RUN_CHAINED"] = True
CONFIG["TRANSFORM"]["OVERWRITE"] = True
CONFIG["TRANSFORM"]["DEBUG"] = False

# GUI
CONFIG["GUI"] = {}
CONFIG["GUI"]["OTC ICON"] = str(
    Path(__file__).parents[0] / r"view" / r"helpers" / r"OTC.ico"
)
CONFIG["GUI"]["FONT"] = "Open Sans"
CONFIG["GUI"]["FONTSIZE"] = 12
CONFIG["GUI"]["WINDOW"] = {}
CONFIG["GUI"]["WINDOW"]["LOCATION_X"] = 0
CONFIG["GUI"]["WINDOW"]["LOCATION_Y"] = 0
CONFIG["GUI"]["FRAMEWIDTH"] = 80
CONFIG["GUI"]["COLWIDTH"] = 50
PAD = {"padx": 10, "pady": 10}


# TODO: #72 Overwrite default config with user config from user.conf (json file)
