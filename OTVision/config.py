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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from OTVision.helpers.log import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)

# CONFIG dict keys
AVAILABLE_WEIGHTS = "AVAILABLEWEIGHTS"
CALIBRATIONS = "CALIBRATIONS"
COL_WIDTH = "COLWIDTH"
CONF = "CONF"
CONVERT = "CONVERT"
DEFAULT_FILETYPE = "DEFAULT_FILETYPE"
DELETE_INPUT = "DELETE_INPUT"
ROTATION = "ROTATION"
DETECT = "DETECT"
DETECTIONS = "DETECTIONS"
FILETYPES = "FILETYPES"
FONT = "FONT"
FONT_SIZE = "FONTSIZE"
FPS_FROM_FILENAME = "FPS_FROM_FILENAME"
FRAME_WIDTH = "FRAMEWIDTH"
GUI = "GUI"
HALF_PRECISION = "HALF_PRECISION"
INPUT_FPS = "INPUT_FPS"
IMG = "IMG"
IMG_SIZE = "IMGSIZE"
IOU = "IOU"
LAST_PATHS = "LAST PATHS"
LOCATION_X = "LOCATION_X"
LOCATION_Y = "LOCATION_Y"
NORMALIZED = "NORMALIZED"
OTC_ICON = "OTC ICON"
OUTPUT_FPS = "OUTPUT_FPS"
OUTPUT_FILETYPE = "OUTPUT_FILETYPE"
OVERWRITE = "OVERWRITE"
PATHS = "PATHS"
RUN_CHAINED = "RUN_CHAINED"
EXPECTED_DURATION = "EXPECTED_DURATION"
REFPTS = "REFPTS"
SEARCH_SUBDIRS = "SEARCH_SUBDIRS"
SIGMA_H = "SIGMA_H"
SIGMA_IOU = "SIGMA_IOU"
SIGMA_L = "SIGMA_L"
T_MIN = "T_MIN"
T_MISS_MAX = "T_MISS_MAX"
TRACK = "TRACK"
TRACKS = "TRACKS"
TRANSFORM = "TRANSFORM"
UNDISTORT = "UNDISTORT"
VID = "VID"
VID_ROTATABLE = "VID_ROTATABLE"
VIDEOS = "VIDEOS"
WEIGHTS = "WEIGHTS"
WINDOW = "WINDOW"
YOLO = "YOLO"
LOG = "LOG"
LOG_LEVEL_CONSOLE = "LOG_LEVEL_CONSOLE"
LOG_LEVEL_FILE = "LOG_LEVEL_FILE"
LOG_DIR = "LOG_DIR"
START_TIME = "START_TIME"
DETECT_END = "DETECT_END"
DETECT_START = "DETECT_START"

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"

"""Default length of a video is 15 minutes."""
DEFAULT_EXPECTED_DURATION: timedelta = timedelta(minutes=15)


@dataclass(frozen=True)
class _LogConfig:
    log_level_console: str = "WARNING"
    log_level_file: str = "DEBUG"

    @staticmethod
    def from_dict(d: dict) -> "_LogConfig":
        return _LogConfig(
            d.get(LOG_LEVEL_CONSOLE, _LogConfig.log_level_console),
            d.get(LOG_LEVEL_FILE, _LogConfig.log_level_file),
        )

    def to_dict(self) -> dict:
        return {
            LOG_LEVEL_CONSOLE: self.log_level_console,
            LOG_LEVEL_FILE: self.log_level_file,
        }


@dataclass(frozen=True)
class _DefaultFiletype:
    video: str = ".mp4"
    image: str = ".jpg"
    detect: str = ".otdet"
    track: str = ".ottrk"
    refpts: str = ".otrfpts"

    @staticmethod
    def from_dict(d: dict) -> "_DefaultFiletype":
        return _DefaultFiletype(
            d.get(VID, _DefaultFiletype.video),
            d.get(IMG, _DefaultFiletype.image),
            d.get(DETECT, _DefaultFiletype.detect),
            d.get(TRACK, _DefaultFiletype.track),
            d.get(REFPTS, _DefaultFiletype.refpts),
        )

    def to_dict(self) -> dict:
        return {
            VID: self.video,
            IMG: self.image,
            DETECT: self.detect,
            TRACK: self.track,
            REFPTS: self.refpts,
        }


@dataclass(frozen=True)
class _VideoFiletypes:
    avi: str = ".avi"
    mkv: str = ".mkv"
    mov: str = ".mov"
    mp4: str = ".mp4"

    def to_list(self) -> list:
        return [
            self.avi,
            self.mkv,
            self.mov,
            self.mp4,
        ]

    def rotatable_to_list(self) -> list:
        return [self.mov, self.mp4]


@dataclass(frozen=True)
class _ImageFiletypes:
    jpg: str = ".jpg"
    jpeg: str = ".jpeg"
    png: str = ".png"

    def to_list(self) -> list:
        return [self.jpg, self.jpeg, self.png]


@dataclass(frozen=True)
class _Filetypes:
    video_filetypes: _VideoFiletypes = _VideoFiletypes()
    image_filetypes: _ImageFiletypes = _ImageFiletypes()
    detect: str = _DefaultFiletype.detect
    track: str = _DefaultFiletype.track
    refpts: str = _DefaultFiletype.refpts
    transform: str = ".gpkg"

    def to_dict(self) -> dict:
        return {
            VID: self.video_filetypes.to_list(),
            VID_ROTATABLE: self.video_filetypes.rotatable_to_list(),
            IMG: self.image_filetypes.to_list(),
            DETECT: [self.detect],
            TRACK: [self.track],
            REFPTS: [self.refpts],
            TRANSFORM: [self.transform],
        }


@dataclass(frozen=True)
class _LastPaths:
    videos: list = field(default_factory=list)
    detections: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    calibrations: list = field(default_factory=list)
    refpts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            VIDEOS: self.videos,
            DETECTIONS: self.detections,
            TRACKS: self.tracks,
            CALIBRATIONS: self.calibrations,
            REFPTS: self.refpts,
        }


@dataclass(frozen=True)
class ConvertConfig:
    paths: list[Path] = field(default_factory=list)
    run_chained: bool = True
    output_filetype: str = _VideoFiletypes.mp4
    input_fps: float = 20.0
    output_fps: float = 20.0
    fps_from_filename: bool = True
    delete_input: bool = False
    rotation: int = 0
    overwrite: bool = True

    @staticmethod
    def from_dict(d: dict) -> "ConvertConfig":
        return ConvertConfig(
            d.get(PATHS, []),
            d.get(RUN_CHAINED, ConvertConfig.run_chained),
            d.get(OUTPUT_FILETYPE, ConvertConfig.output_filetype),
            d.get(INPUT_FPS, ConvertConfig.input_fps),
            d.get(OUTPUT_FPS, ConvertConfig.output_fps),
            d.get(FPS_FROM_FILENAME, ConvertConfig.fps_from_filename),
            d.get(DELETE_INPUT, ConvertConfig.delete_input),
            d.get(ROTATION, ConvertConfig.rotation),
            d.get(OVERWRITE, ConvertConfig.overwrite),
        )

    def to_dict(self) -> dict:
        return {
            PATHS: self.paths,
            RUN_CHAINED: self.run_chained,
            OUTPUT_FILETYPE: self.output_filetype,
            INPUT_FPS: self.input_fps,
            OUTPUT_FPS: self.output_fps,
            FPS_FROM_FILENAME: self.fps_from_filename,
            DELETE_INPUT: self.delete_input,
            ROTATION: self.rotation,
            OVERWRITE: self.overwrite,
        }


@dataclass(frozen=True)
class _YoloWeights:
    yolov8s: str = "yolov8s"
    yolov8m: str = "yolov8m"
    yolov8l: str = "yolov8l"
    yolov8x: str = "yolov8x"

    def to_list(self) -> list:
        return [self.yolov8s, self.yolov8m, self.yolov8l, self.yolov8x]


@dataclass(frozen=True)
class YoloConfig:
    """Represents the configuration for the YOLO model.

    Attributes:
        weights (str): Path to YOLO model weights.
        available_weights (_YoloWeights): List of available default YOLO model weights.
        conf (float): Confidence threshold.
        iou (float): Intersection over union threshold.
        img_size (int): Size of the input image.
        chunk_size (int): Chunk size for processing.
        normalized (bool): Whether to normalize the bounding boxes.
    """

    weights: str = _YoloWeights.yolov8s
    available_weights: _YoloWeights = _YoloWeights()
    conf: float = 0.25
    iou: float = 0.45
    img_size: int = 640
    chunk_size: int = 1
    normalized: bool = False

    @staticmethod
    def from_dict(d: dict) -> "YoloConfig":
        return YoloConfig(
            weights=d.get(WEIGHTS, YoloConfig.weights),
            conf=d.get(CONF, YoloConfig.conf),
            iou=d.get(IOU, YoloConfig.iou),
            img_size=d.get(IMG_SIZE, YoloConfig.img_size),
            normalized=d.get(NORMALIZED, YoloConfig.normalized),
        )

    def to_dict(self) -> dict:
        return {
            WEIGHTS: self.weights,
            AVAILABLE_WEIGHTS: self.available_weights.to_list(),
            CONF: self.conf,
            IOU: self.iou,
            IMG_SIZE: self.img_size,
            NORMALIZED: self.normalized,
        }


@dataclass(frozen=True)
class DetectConfig:
    """Represents the configuration for the `detect` command.

    Attributes:
        paths (list[Path]): List of  files to be processed.
        run_chained (bool): Whether to run chained commands.
        yolo_config (YoloConfig): Configuration for the YOLO model.
        expected_duration (timedelta | None): Expected duration of the video.
            `None` if unknown.
        overwrite (bool): Whether to overwrite existing files.
        half_precision (bool): Whether to use half precision.
        detect_start (int | None): Start frame for detection expressed in seconds.
            Value `None` marks the start of the video.
        detect_end (int | None): End frame for detection expressed in seconds.
            Value `None` marks the end of the video.

    """

    @property
    def confidence(self) -> float:
        """Gets the confidence level set in the YOLO configuration.

        Returns:
            float: The intersection over union threshold value.
        """
        return self.yolo_config.conf

    @property
    def weights(self) -> str:
        return self.yolo_config.weights

    @property
    def iou(self) -> float:
        return self.yolo_config.iou

    @property
    def img_size(self) -> int:
        return self.yolo_config.img_size

    @property
    def normalized(self) -> bool:
        return self.yolo_config.normalized

    paths: list[Path] = field(default_factory=list)
    run_chained: bool = True
    yolo_config: YoloConfig = YoloConfig()
    expected_duration: timedelta | None = None
    overwrite: bool = True
    half_precision: bool = False
    start_time: datetime | None = None
    detect_start: int | None = None
    detect_end: int | None = None

    @staticmethod
    def from_dict(d: dict) -> "DetectConfig":
        yolo_config_dict = d.get(YOLO)
        yolo_config = (
            YoloConfig.from_dict(yolo_config_dict)
            if yolo_config_dict
            else DetectConfig.yolo_config
        )

        files = [Path(file).expanduser() for file in d.get(PATHS, [])]
        expected_duration = d.get(EXPECTED_DURATION, None)
        if expected_duration is not None:
            expected_duration = timedelta(seconds=int(expected_duration))

        start_time = DetectConfig._parse_start_time(d)
        return DetectConfig(
            files,
            d.get(RUN_CHAINED, DetectConfig.run_chained),
            yolo_config,
            expected_duration,
            d.get(OVERWRITE, DetectConfig.overwrite),
            d.get(HALF_PRECISION, DetectConfig.half_precision),
            start_time,
            d.get(DETECT_START, DetectConfig.detect_start),
            d.get(DETECT_END, DetectConfig.detect_end),
        )

    @staticmethod
    def _parse_start_time(d: dict) -> datetime | None:
        if start_time := d.get(START_TIME, DetectConfig.start_time):
            return datetime.strptime(start_time, DATETIME_FORMAT)
        return start_time

    def to_dict(self) -> dict:
        expected_duration = (
            int(self.expected_duration.total_seconds())
            if self.expected_duration is not None
            else None
        )
        return {
            PATHS: [str(p) for p in self.paths],
            RUN_CHAINED: self.run_chained,
            YOLO: self.yolo_config.to_dict(),
            EXPECTED_DURATION: expected_duration,
            OVERWRITE: self.overwrite,
            HALF_PRECISION: self.half_precision,
            START_TIME: self.start_time,
            DETECT_START: self.detect_start,
            DETECT_END: self.detect_end,
        }


@dataclass(frozen=True)
class _TrackIouConfig:
    sigma_l: float = 0.27
    sigma_h: float = 0.42
    sigma_iou: float = 0.38
    t_min: int = 5
    t_miss_max: int = 51

    @staticmethod
    def from_dict(d: dict) -> "_TrackIouConfig":
        return _TrackIouConfig(
            d.get(SIGMA_L, _TrackIouConfig.sigma_l),
            d.get(SIGMA_H, _TrackIouConfig.sigma_h),
            d.get(SIGMA_IOU, _TrackIouConfig.sigma_iou),
            d.get(T_MIN, _TrackIouConfig.t_min),
            d.get(T_MISS_MAX, _TrackIouConfig.t_miss_max),
        )

    def to_dict(self) -> dict:
        return {
            SIGMA_L: self.sigma_l,
            SIGMA_H: self.sigma_h,
            SIGMA_IOU: self.sigma_iou,
            T_MIN: self.t_min,
            T_MISS_MAX: self.t_miss_max,
        }


@dataclass(frozen=True)
class TrackConfig:
    paths: list[Path] = field(default_factory=list)
    run_chained: bool = True
    iou: _TrackIouConfig = _TrackIouConfig()
    overwrite: bool = True

    @staticmethod
    def from_dict(d: dict) -> "TrackConfig":
        iou_config_dict = d.get(IOU)
        iou_config = (
            _TrackIouConfig.from_dict(iou_config_dict)
            if iou_config_dict
            else TrackConfig.iou
        )

        return TrackConfig(
            d.get(PATHS, []),
            d.get(RUN_CHAINED, TrackConfig.run_chained),
            iou_config,
            d.get(OVERWRITE, TrackConfig.overwrite),
        )

    def to_dict(self) -> dict:
        return {
            PATHS: [str(p) for p in self.paths],
            RUN_CHAINED: self.run_chained,
            IOU: self.iou.to_dict(),
            OVERWRITE: self.overwrite,
        }


@dataclass(frozen=True)
class _UndistortConfig:
    overwrite: bool = False

    @staticmethod
    def from_dict(d: dict) -> "_UndistortConfig":
        return _UndistortConfig(
            d.get(OVERWRITE, _UndistortConfig.overwrite),
        )

    def to_dict(self) -> dict:
        return {OVERWRITE: self.overwrite}


@dataclass(frozen=True)
class _TransformConfig:
    paths: list[Path] = field(default_factory=list)
    run_chained: bool = True
    overwrite: bool = True

    @staticmethod
    def from_dict(d: dict) -> "_TransformConfig":
        return _TransformConfig(
            d.get(PATHS, []),
            d.get(RUN_CHAINED, _TransformConfig.run_chained),
            d.get(OVERWRITE, _TransformConfig.overwrite),
        )

    def to_dict(self) -> dict:
        return {
            PATHS: [str(p) for p in self.paths],
            RUN_CHAINED: self.run_chained,
            OVERWRITE: self.overwrite,
        }


@dataclass(frozen=True)
class _GuiWindowConfig:
    location_x: int = 0
    location_y: int = 0

    @staticmethod
    def from_dict(d: dict) -> "_GuiWindowConfig":
        return _GuiWindowConfig(
            d.get(LOCATION_X, _GuiWindowConfig.location_x),
            d.get(LOCATION_Y, _GuiWindowConfig.location_y),
        )

    def to_dict(self) -> dict:
        return {LOCATION_X: self.location_x, LOCATION_Y: self.location_y}


@dataclass(frozen=True)
class _GuiConfig:
    otc_icon: str = str(Path(__file__).parents[0] / r"view" / r"helpers" / r"OTC.ico")
    font: str = "Open Sans"
    font_size: int = 12
    window_config: _GuiWindowConfig = _GuiWindowConfig()
    frame_width: int = 80
    col_width: int = 50

    @staticmethod
    def from_dict(d: dict) -> "_GuiConfig":
        window_config_dict = d.get(WINDOW)
        window_config = (
            _GuiWindowConfig.from_dict(window_config_dict)
            if window_config_dict
            else _GuiConfig.window_config
        )

        return _GuiConfig(
            font=d.get(FONT, _GuiConfig.font),
            font_size=d.get(FONT_SIZE, _GuiConfig.font_size),
            window_config=window_config,
            frame_width=d.get(FRAME_WIDTH, _GuiConfig.frame_width),
            col_width=d.get(COL_WIDTH, _GuiConfig.col_width),
        )

    def to_dict(self) -> dict:
        return {
            OTC_ICON: self.otc_icon,
            FONT: self.font,
            FONT_SIZE: self.font_size,
            WINDOW: self.window_config.to_dict(),
            FRAME_WIDTH: self.frame_width,
            COL_WIDTH: self.col_width,
        }


@dataclass
class Config:
    """Represents the OTVision config file.

    Provides methods to parse in a custom config file from a dict or a YAML file.
    Updates the default configuration with the custom config.
    """

    log: _LogConfig = _LogConfig()
    search_subdirs: bool = True
    default_filetype: _DefaultFiletype = _DefaultFiletype()
    filetypes: _Filetypes = _Filetypes()
    last_paths: _LastPaths = _LastPaths()
    convert: ConvertConfig = ConvertConfig()
    detect: DetectConfig = DetectConfig()
    track: TrackConfig = TrackConfig()
    undistort: _UndistortConfig = _UndistortConfig()
    transform: _TransformConfig = _TransformConfig()
    gui: _GuiConfig = _GuiConfig()

    @staticmethod
    def from_dict(d: dict) -> "Config":
        """Builds a OTVision `Config` object with a dictionary.

        Args:
            d (dict): The dictionary to be parsed.

        Returns:
            Config: The built `Config` object.
        """
        log_dict = d.get(LOG)
        default_filtetype_dict = d.get(DEFAULT_FILETYPE)
        convert_dict = d.get(CONVERT)
        detect_dict = d.get(DETECT)
        track_dict = d.get(TRACK)
        undistort_dict = d.get(UNDISTORT)
        transform_dict = d.get(TRANSFORM)
        gui_dict = d.get(GUI)

        log_config = _LogConfig.from_dict(log_dict) if log_dict else Config.log
        default_filetype = (
            _DefaultFiletype.from_dict(default_filtetype_dict)
            if default_filtetype_dict
            else Config.default_filetype
        )
        convert_config = (
            ConvertConfig.from_dict(convert_dict) if convert_dict else Config.convert
        )
        detect_config = (
            DetectConfig.from_dict(detect_dict) if detect_dict else Config.detect
        )
        track_config = TrackConfig.from_dict(track_dict) if track_dict else Config.track
        undistort_config = (
            _UndistortConfig.from_dict(undistort_dict)
            if undistort_dict
            else Config.undistort
        )
        transform_config = (
            _TransformConfig.from_dict(transform_dict)
            if transform_dict
            else Config.transform
        )
        gui_config = _GuiConfig.from_dict(gui_dict) if gui_dict else Config.gui

        return Config(
            log=log_config,
            search_subdirs=d.get(SEARCH_SUBDIRS, Config.search_subdirs),
            default_filetype=default_filetype,
            convert=convert_config,
            detect=detect_config,
            track=track_config,
            undistort=undistort_config,
            transform=transform_config,
            gui=gui_config,
        )

    def to_dict(self) -> dict:
        """Returns the OTVision config as a dict.

        Returns:
            dict: The OTVision config.
        """
        return {
            LOG: self.log.to_dict(),
            SEARCH_SUBDIRS: self.search_subdirs,
            DEFAULT_FILETYPE: self.default_filetype.to_dict(),
            FILETYPES: self.filetypes.to_dict(),
            LAST_PATHS: self.last_paths.to_dict(),
            CONVERT: self.convert.to_dict(),
            DETECT: self.detect.to_dict(),
            TRACK: self.track.to_dict(),
            UNDISTORT: self.undistort.to_dict(),
            TRANSFORM: self.transform.to_dict(),
            GUI: self.gui.to_dict(),
        }

    @staticmethod
    def from_yaml(yaml_file: Path) -> dict:
        """Parse OTVision yaml configuration file.

        Args:
            yaml_file (Path): The yaml config file.

        Returns:
            dict: The parsed config file as a dict.
        """
        with open(yaml_file, "r") as file:
            try:
                yaml_config = yaml.safe_load(file)
            except yaml.YAMLError:
                log.exception("Unable to parse user config. Using default config.")
                raise
        config = Config.from_dict(yaml_config)

        return config.to_dict()


class ConfigParser:
    def parse(self, config_file: Path) -> Config:
        """Parse OTVision yaml configuration file.

        Args:
            config_file (Path): The yaml config file.

        Returns:
            Config: The parsed config file.
        """
        with open(config_file, "r") as file:
            try:
                yaml_config = yaml.safe_load(file)
            except yaml.YAMLError:
                log.exception("Unable to parse user config. Using default config.")
                raise
        return Config.from_dict(yaml_config)


def parse_user_config(yaml_file: Path | str) -> Config:
    """Parses a custom OTVision user config yaml file.

    Args:
        yaml_file (Path |str): The absolute Path to the config file.
    """
    user_config_file = Path(yaml_file)
    user_config = ConfigParser().parse(user_config_file)
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
