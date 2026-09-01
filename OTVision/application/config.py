from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from OTVision.domain.tracker import TrackerLifecycle, TrackerType
from OTVision.plugin.ffmpeg_video_writer import (
    ConstantRateFactor,
    EncodingSpeed,
    VideoCodec,
)

# Scalar values accepted in TRACK.BOT_SORT tracker_params / Ultralytics args.
BotSortTrackerParam = bool | int | float | str

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
BOT_SORT = "BOT_SORT"
TRACKER_TYPE = "TRACKER_TYPE"
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
WRITE_VIDEO = "WRITE_VIDEO"
VIDEO_CODEC = "VIDEO_CODEC"
ENCODING_SPEED = "ENCODING_SPEED"
CRF = "CRF"
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
DEFAULT_EXPECTED_DURATION: timedelta = timedelta(minutes=15)
"""Default length of a video is 15 minutes."""
STREAM = "STREAM"
STREAM_SAVE_DIR = "SAVE_DIR"
STREAM_NAME = "NAME"
STREAM_SOURCE = "SOURCE"
FLUSH_BUFFER_SIZE = "FLUSH_BUFFER_SIZE"


@dataclass(frozen=True)
class _LogConfig:
    log_level_console: str = "WARNING"
    log_level_file: str = "DEBUG"

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
    paths: list[str] = field(default_factory=list)
    run_chained: bool = True
    output_filetype: str = _VideoFiletypes.mp4
    input_fps: float = 20.0
    output_fps: float = 20.0
    fps_from_filename: bool = True
    delete_input: bool = False
    rotation: int = 0
    overwrite: bool = True

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

    paths: list[str] = field(default_factory=list)
    run_chained: bool = True
    yolo_config: YoloConfig = YoloConfig()
    expected_duration: timedelta | None = None
    overwrite: bool = True
    half_precision: bool = False
    start_time: datetime | None = None
    detect_start: int | None = None
    detect_end: int | None = None
    write_video: bool = False
    video_codec: VideoCodec = VideoCodec.H264_SOFTWARE
    encoding_speed: EncodingSpeed = EncodingSpeed.FAST
    crf: ConstantRateFactor = ConstantRateFactor.DEFAULT

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
            WRITE_VIDEO: self.write_video,
            VIDEO_CODEC: self.video_codec.value,
            ENCODING_SPEED: self.encoding_speed.value,
            CRF: self.crf.name,
        }


@dataclass(frozen=True)
class _TrackIouConfig:
    sigma_l: float = 0.27
    sigma_h: float = 0.42
    sigma_iou: float = 0.38
    t_min: int = 5
    t_miss_max: int = 51

    def to_dict(self) -> dict:
        return {
            SIGMA_L: self.sigma_l,
            SIGMA_H: self.sigma_h,
            SIGMA_IOU: self.sigma_iou,
            T_MIN: self.t_min,
            T_MISS_MAX: self.t_miss_max,
        }


# Tuned continuous-tracking defaults for OTCamera .otdet input (typically 20 fps).
# `track_buffer` is intentionally omitted: when unset, BotsortTracker derives it as
# ceil(t_miss_max * 30 / fps) so Ultralytics' occlusion window is at least T_MISS_MAX.
DEFAULT_BOTSORT_TRACKER_PARAMS: dict[str, BotSortTrackerParam] = {
    "track_high_thresh": 0.2,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.2,
    "match_thresh": 0.9,
    "fuse_score": True,
    "gmc_method": "none",
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.25,
    "with_reid": False,
}


# `track_buffer` is a valid override even though it is deliberately absent from
# the defaults: when unset, BotsortTracker derives it from T_MISS_MAX and FPS.
ALLOWED_BOTSORT_TRACKER_PARAMS: frozenset[str] = frozenset(
    DEFAULT_BOTSORT_TRACKER_PARAMS
) | {"track_buffer"}


@dataclass(frozen=True)
class _TrackBotSortConfig:
    """BoT-SORT tracker configuration.

    We keep `t_min`/`t_miss_max` as our pipeline lifecycle parameters, while
    all other keys are forwarded to ultralytics' BoT-SORT implementation.
    """

    t_min: int = 5
    t_miss_max: int = 60
    tracker_params: dict[str, BotSortTrackerParam] = field(
        default_factory=lambda: dict(DEFAULT_BOTSORT_TRACKER_PARAMS)
    )

    def to_dict(self) -> dict[str, BotSortTrackerParam]:
        """Serialize BoT-SORT config including forwarded tracker params.

        Returns:
            dict[str, BotSortTrackerParam]: Mapping for YAML / config round-trips.
        """
        merged: dict[str, BotSortTrackerParam] = {
            T_MIN: self.t_min,
            T_MISS_MAX: self.t_miss_max,
        }
        merged.update(self.tracker_params)
        return merged


@dataclass(frozen=True)
class TrackConfig:
    def __post_init__(self) -> None:
        """Normalize ``tracker_type`` to a :class:`TrackerType` member.

        ``StrEnum`` members compare equal to their string value but are not
        identical to it, so a config built with a plain ``"botsort"`` would
        silently take the IOU branch of every ``is`` comparison. Coercing once
        here keeps that invariant true however the config was constructed.

        Raises:
            ValueError: If ``tracker_type`` is not a known tracker.
        """
        if not isinstance(self.tracker_type, TrackerType):
            object.__setattr__(
                self, "tracker_type", TrackerType(str(self.tracker_type).lower())
            )

    @property
    def sigma_l(self) -> float:
        return self.iou.sigma_l

    @property
    def sigma_h(self) -> float:
        return self.iou.sigma_h

    @property
    def sigma_iou(self) -> float:
        return self.iou.sigma_iou

    @property
    def lifecycle(self) -> TrackerLifecycle:
        """Lifecycle thresholds of the selected tracker.

        This is the single deliberate dispatch point on tracker type. Code that
        is specific to one tracker must read that tracker's own config instead.

        Returns:
            TrackerLifecycle: Thresholds of the tracker named by ``tracker_type``.
        """
        if self.tracker_type is TrackerType.BOTSORT:
            return TrackerLifecycle(self.botsort.t_min, self.botsort.t_miss_max)
        return TrackerLifecycle(self.iou.t_min, self.iou.t_miss_max)

    paths: list[str] = field(default_factory=list)
    run_chained: bool = True
    iou: _TrackIouConfig = _TrackIouConfig()
    tracker_type: TrackerType = TrackerType.IOU
    botsort: _TrackBotSortConfig = field(default_factory=_TrackBotSortConfig)
    overwrite: bool = True

    def to_dict(self) -> dict:
        return {
            PATHS: [str(p) for p in self.paths],
            RUN_CHAINED: self.run_chained,
            IOU: self.iou.to_dict(),
            BOT_SORT: self.botsort.to_dict(),
            TRACKER_TYPE: self.tracker_type.value,
            OVERWRITE: self.overwrite,
        }


@dataclass(frozen=True)
class _UndistortConfig:
    overwrite: bool = False

    def to_dict(self) -> dict:
        return {OVERWRITE: self.overwrite}


@dataclass(frozen=True)
class _TransformConfig:
    paths: list[str] = field(default_factory=list)
    run_chained: bool = True
    overwrite: bool = True

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

    def to_dict(self) -> dict:
        return {
            OTC_ICON: self.otc_icon,
            FONT: self.font,
            FONT_SIZE: self.font_size,
            WINDOW: self.window_config.to_dict(),
            FRAME_WIDTH: self.frame_width,
            COL_WIDTH: self.col_width,
        }


@dataclass(frozen=True)
class StreamConfig:
    name: str
    source: str
    save_dir: Path
    flush_buffer_size: int

    def to_dict(self) -> dict:
        return {
            STREAM_NAME: self.name,
            STREAM_SOURCE: self.source,
            STREAM_SAVE_DIR: str(self.save_dir),
            FLUSH_BUFFER_SIZE: self.flush_buffer_size,
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
    stream: StreamConfig | None = None

    def to_dict(self) -> dict:
        """Returns the OTVision config as a dict.

        Returns:
            dict: The OTVision config.
        """
        data = {
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
        if self.stream is not None:
            data[STREAM] = self.stream.to_dict()
        return data
