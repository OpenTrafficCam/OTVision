from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


class CliArgs(ABC):
    @abstractmethod
    def get_config_file(self) -> Path | None:
        raise NotImplementedError


class CliParseError(Exception):
    pass


@dataclass
class DetectCliArgs(CliArgs):
    expected_duration: timedelta | None
    paths: list[Path] | None
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

    def get_config_file(self) -> Path | None:
        return self.config_file


class DetectCliParser(ABC):
    @abstractmethod
    def parse(self) -> DetectCliArgs:
        raise NotImplementedError
