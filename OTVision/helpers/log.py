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
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(level=logging.WARNING)
log.addHandler(hdlr=console_handler)

# Create file handler
datetime_str = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
log_dir = Path.cwd() / "otvision_logs"
if not log_dir.is_dir():
    log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"{datetime_str}.log"
file_handler = logging.FileHandler(filename=log_file)
file_handler.setLevel(level=logging.DEBUG)
log.addHandler(hdlr=file_handler)


def set_debug() -> None:
    """Sets logging level to DEBUG"""
    log.setLevel("DEBUG")
    log.info("Debug mode on")


def reset_debug() -> None:
    """Resets logging level to INFO"""
    log.info("Debug mode off")
    log.setLevel("INFO")
