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

LOGGER_NAME = "OTVision Logger"

DEFAULT_DIR = Path.cwd()

VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class _OTVisionLogger:
    """Class for creating a logging.Logger.
    Should only be instanciated once in the same module as this class.
    To access this instance, use logging.getLogger(LOGGER_NAME)
    with LOGGER_NAME from the same module where this class is defined.
    """

    def __init__(self, name: str = LOGGER_NAME) -> None:
        self.logger = logging.getLogger(name=name)
        self.logger.setLevel("DEBUG")
        self._set_filename()
        self._set_formatter()

    def _set_formatter(self) -> None:
        self.formatter = logging.Formatter(
            "%(asctime)s %(levelname)s (%(filename)s::%(funcName)s"
            "::%(lineno)d): %(message)s"
        )

    def _set_filename(self) -> None:
        datetime_str = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
        self.filename = f"{datetime_str}.log"

    def _add_handler(self, handler: logging.Handler, level: str) -> None:
        handler.setFormatter(self.formatter)
        handler.setLevel(level=level)
        self.logger.addHandler(handler)

    def add_file_handler(
        self, log_dir: Path = DEFAULT_DIR, level: str = "DEBUG"
    ) -> None:
        """Add a file handler to the already existing global instance of
        _OTVisionLogger.
        Should only be used once in each of OTVisions command line or
        graphical user interfaces.

        Args:
            log_dir (Path): Path to the directory to write the logs.
                Defaults to None.
            level (str): Logging level of the file handler.
                One from "DEBUG", "INFO", "WARNING", "ERROR" or "CRITICAL".

        IMPORTANT:
            log_dir and level are not intended to be optional, they have to be provided
            in every case. The default values provided are a safety net.
        """
        log_subdir = log_dir / "_otvision_logs"
        if not log_subdir.is_dir():
            log_subdir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_subdir / self.filename)
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


# This here should be the only time the _OTVisionLogger is "directly" instanciated
# In all other module that should be logged from, use logging.getLogger(LOGGER_NAME)

log = _OTVisionLogger()
