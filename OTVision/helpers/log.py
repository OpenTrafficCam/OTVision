"""
OTVision helpers for logging
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
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

LOGGER_NAME = "OTVision Logger"

DEFAULT_LOG_NAME = f"{datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')}"
LOG_EXT = "log"
DEFAULT_LOG_FILE = Path(f"logs/{DEFAULT_LOG_NAME}.{LOG_EXT}")

VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

LOG_FORMAT: str = (
    "%(asctime)s %(levelname)s (%(filename)s::%(funcName)s::%(lineno)d): %(message)s"
)

LOG_LEVEL_INTEGERS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


class LogFileAlreadyExists(Exception):
    pass


class _OTVisionLogger:
    """Class for creating a logging.Logger.
    Should only be instantiated once in the same module as this class.
    To access this instance, use logging.getLogger(LOGGER_NAME)
    with LOGGER_NAME from the same module where this class is defined.
    """

    def __init__(
        self,
        name: str = LOGGER_NAME,
        datetime_provider: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._provide_datetime = datetime_provider
        self.logger = logging.getLogger(name=name)
        self.logger.setLevel("DEBUG")
        self._set_formatter()

    def _set_formatter(self) -> None:
        self.formatter = logging.Formatter(LOG_FORMAT)

    def _add_handler(self, handler: logging.Handler, level: str) -> None:
        handler.setFormatter(self.formatter)
        handler.setLevel(level=level)
        self.logger.addHandler(handler)

    def add_file_handler(
        self,
        log_file: Path = DEFAULT_LOG_FILE,
        level: str = "DEBUG",
        overwrite: bool = False,
    ) -> None:
        """Add a file handler to the already existing global instance of
        _OTVisionLogger.

        Should only be used once in each of OTVisions command line or
        graphical user interfaces.

        Args:
            log_file (Path): file path to write the logs. Defaults to None.
            level (str): Logging level of the file handler.
                One from "DEBUG", "INFO", "WARNING", "ERROR" or "CRITICAL".
            overwrite (bool): if True, overwrite existing log file. Defaults to False.

        IMPORTANT:
            log_file and level are not intended to be optional, they have to be provided
            in every case. The default values provided are a safety net.
        """
        if log_file.is_file() and not overwrite:
            raise LogFileAlreadyExists(
                f"Log file '{log_file}' already exists. "
                "Please specify option to overwrite the log file when using the CLI."
            )
        log_dir = log_file.parent
        if log_file.is_dir() or not log_file.suffix:
            log_dir = log_file / "logs"
            log_file = (
                log_dir
                / f"{self._provide_datetime().strftime(r'%Y-%m-%d_%H-%M-%S')}.{LOG_EXT}"
            )

        log_dir.mkdir(parents=True, exist_ok=True)
        log_file.touch()
        file_handler = logging.FileHandler(log_file, mode="w")
        self._add_handler(file_handler, level)

    def add_console_handler(self, level: str = "WARNING") -> None:
        """Add a console handler to the already existing global instance of
        _OTVisionLogger.
        Should only be used once in each of OTVisions command line or
        graphical user interfaces.

        Args:
            level (str): Logging level of the console handler.
                One from "DEBUG", "INFO", "WARNING", "ERROR" or "CRITICAL".
                Defaults to "WARNING".

        IMPORTANT:
            level is not intended to be optional, it has to be provided
            in every case. The default value provided is a safety net.
        """
        console_handler = logging.StreamHandler(sys.stdout)
        self._add_handler(console_handler, level)

    def _remove_handlers(self) -> None:
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)


# This here should be the only time the _OTVisionLogger is "directly" instantiated
# In all other module that should be logged from, use logging.getLogger(LOGGER_NAME)

log = _OTVisionLogger()
