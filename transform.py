"""
OTVision script to call the transform main with arguments parsed from command line
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

from OTVision.helpers.log import log
from OTVision.transform.transform import main as transform


def parse():
    parser = argparse.ArgumentParser(description="Track objects in detections")
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        type=str,
        help="Path or list of paths to tracks files",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--refpts_file",
        type=str,
        help="Path to refpts file. If not given, each tracks file should have one",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Logging in debug mode"
    )
    return parser.parse_args()


def main():
    kwargs = vars(parse())
    log.info("Starting transforming to world coordinates from command line")
    log.info(f"Arguments: {kwargs}")
    transform(**kwargs)
    log.info("Finished transforming to world coordinates  from command line")


if __name__ == "__main__":
    main()
