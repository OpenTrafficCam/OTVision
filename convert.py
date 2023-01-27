"""
OTVision script to call the convert main with arguments parsed from command line
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
from pathlib import Path

from OTVision.config import CONFIG
from OTVision.convert.convert import main as convert
from OTVision.helpers.log import log


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert h264 to mp4")
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        type=str,
        help="Path or list of paths to h264 (or other) video files",
        required=True,
    )
    parser.add_argument(
        "--delete_input",
        default=CONFIG["CONVERT"]["DELETE_INPUT"],
        action=argparse.BooleanOptionalAction,
        help="Delete input files after convert",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        default=CONFIG["CONVERT"]["OVERWRITE"],
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "-d",
        "--debug",
        default=CONFIG["CONVERT"]["DEBUG"],
        action=argparse.BooleanOptionalAction,
        help="Logging in debug mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse()
    paths = [Path(str_path) for str_path in args.paths]
    log.info("Starting conversion from command line")
    log.info(f"Arguments: {vars(args)}")
    convert(
        paths=paths,
        delete_input=args.delete_input,
        overwrite=args.overwrite,
        debug=args.debug,
    )
    log.info("Finished conversion from command line")


if __name__ == "__main__":
    main()
