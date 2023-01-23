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

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(message)s", level=logging.INFO
)

log = logging.getLogger(__name__)


def set_debug():
    """Sets logging level to DEBUG"""
    log.setLevel("DEBUG")
    log.debug("Debug mode on")


def reset_debug():
    """Resets logging level to INFO"""
    log.debug("Debug mode off")
    log.setLevel("INFO")
