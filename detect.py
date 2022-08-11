"""
OTVision script to call the detect main with arguments parsed from command line 
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


import argparse

from OTVision.detect.detect import main as detect
from OTVision.helpers.log import log


def parse():
    parser = argparse.ArgumentParser(description="Detect objects in videos or images")
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        type=str,
        help="Path or list of paths to image or video or folder containing videos/images",
        required=True,
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Logging in debug mode"
    )
    return parser.parse_args()


def main():
    kwargs = vars(parse())
    log.info("Starting detection from command line")
    log.info(f"Arguments: {kwargs}")
    detect(**kwargs)
    log.info("Finished detection from command line")


if __name__ == "__main__":
    main()
