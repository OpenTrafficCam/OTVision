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

FORMATTER = logging.Formatter(
    "%(asctime)s %(levelname)s (%(name)s in %(funcName)s"
    " at line %(lineno)d): %(message)s"
)

DATETIME_STR = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")

LOG_DIR = Path.cwd() / "otvision_logs"

LOG_FILENAME = f"{DATETIME_STR}.log"


def get_log_file() -> Path:
    if not LOG_DIR.is_dir():
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / LOG_FILENAME


FILE_HANDLER: logging.Handler = logging.FileHandler(filename=get_log_file())
FILE_HANDLER.setFormatter(FORMATTER)
FILE_HANDLER.setLevel(level=logging.DEBUG)


def get_console_handler() -> logging.Handler:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    console_handler.setLevel(level=logging.WARNING)
    return console_handler


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(hdlr=get_console_handler())
    logger.addHandler(hdlr=FILE_HANDLER)
    logger.propagate = False
    return logger
