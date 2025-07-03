from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from OTVision.plugin.ffmpeg_video_writer import (
    ConstantRateFactor,
    EncodingSpeed,
    VideoCodec,
)


class CliArgs(ABC):
    @abstractmethod
    def get_config_file(self) -> Path | None:
        raise NotImplementedError


class CliParseError(Exception):
    pass


@dataclass
class DetectCliArgs(CliArgs):
    expected_duration: timedelta | None
    paths: list[str] | None
    config_file: Path | None
    logfile: Path
    logfile_overwrite: bool
    log_level_console: str | None
    log_level_file: str | None
    weights: str | None = None
    conf: float | None = None
    iou: float | None = None
    imagesize: int | None = None
    half: bool | None = None
    overwrite: bool | None = None
    start_time: datetime | None = None
    detect_start: int | None = None
    detect_end: int | None = None
    write_video: bool | None = None
    video_codec: VideoCodec | None = None
    encoding_speed: EncodingSpeed | None = None
    crf: ConstantRateFactor | None = None

    def get_config_file(self) -> Path | None:
        return self.config_file


class DetectCliParser(ABC):
    @abstractmethod
    def parse(self) -> DetectCliArgs:
        raise NotImplementedError


@dataclass
class TrackCliArgs(CliArgs):
    paths: list[str] | None
    config_file: Path | None
    logfile: Path
    logfile_overwrite: bool
    log_level_console: str | None
    log_level_file: str | None
    overwrite: bool | None = None
    sigma_l: float | None = None
    sigma_h: float | None = None
    sigma_iou: float | None = None
    t_min: int | None = None
    t_miss_max: int | None = None

    def get_config_file(self) -> Path | None:
        return self.config_file


class TrackCliParser(ABC):
    @abstractmethod
    def parse(self) -> TrackCliArgs:
        """Parse track CLI arguments.

        Returns:
            TrackCliArgs: the parsed track CLI arguments.
        """
        raise NotImplementedError
