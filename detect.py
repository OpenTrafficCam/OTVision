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
from pathlib import Path

from OTVision.config import CONFIG
from OTVision.detect.detect import main as detect
from OTVision.helpers.log import log


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect objects in videos or images")
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        type=str,
        help="Path/list of paths to image or video or folder containing videos/images",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--filetypes",
        default=CONFIG["FILETYPES"]["VID_IMG"],
        type=str,
        nargs="+",
        help="Filetypes of files in folders to select for detection",
        required=False,
    )
    parser.add_argument(
        "-w",
        "--weights",
        default=CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
        type=str,
        help="Name of weights from PyTorch hub or Path to weights file",
        required=False,
    )
    parser.add_argument(
        "-n",
        "--no_overwrite",
        default=CONFIG["DETECT"]["OVERWRITE"],
        action="store_true",
        help="Do not overwrite existing output files",
    )
    parser.add_argument(
        "-d",
        "--debug",
        default=CONFIG["DETECT"]["DEBUG"],
        action="store_true",
        help="Logging in debug mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse()
    paths = [Path(str_path) for str_path in args.paths]
    overwrite = not args.no_overwrite
    log.info("Starting detection from command line")
    log.info(f"Arguments: {vars(args)}")
    detect(
        paths=paths,
        weights=args.weights,
        filetypes=args.filetypes,
        overwrite=overwrite,
        debug=args.debug,
    )
    log.info("Finished detection from command line")


if __name__ == "__main__":
    main()
