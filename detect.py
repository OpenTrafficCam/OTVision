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

import OTVision
import OTVision.config as config
from OTVision.helpers.log import log


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect objects in videos or images")
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        type=str,
        help="Path/list of paths to image or video or folder containing videos/images",
        required=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to custom user configuration yaml file.",
        required=False,
    )
    parser.add_argument(
        "-f",
        "--filetypes",
        type=str,
        nargs="+",
        help="Filetypes of files in folders to select for detection",
        required=False,
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        help="Name of weights from PyTorch hub or Path to weights file",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action=argparse.BooleanOptionalAction,
        help="Logging in debug mode",
    )
    return parser.parse_args()


def _process_config(args: argparse.Namespace) -> None:
    if args.config:
        config.parse_user_config(args.config)
    else:
        user_config_cwd = Path(__file__).parent / "user_config.otvision.yaml"

        if user_config_cwd.exists():
            config.parse_user_config(str(user_config_cwd))


def _extract_paths(args: argparse.Namespace) -> list[str]:
    if args.paths:
        return args.paths
    if len(config.CONFIG["DETECT"]["PATHS"]) == 0:
        raise IOError(
            "No paths have been passed as command line args."
            "No paths have been defined in the user config."
        )

    return config.CONFIG["DETECT"]["PATHS"]


def main() -> None:  # sourcery skip: assign-if-exp
    args = parse()
    _process_config(args)

    try:
        str_paths = _extract_paths(args)
    except IOError as ioe:
        log.error(ioe)

    paths = [Path(str_path) for str_path in str_paths]

    if args.weights is None:
        weights = config.CONFIG["DETECT"]["YOLO"]["WEIGHTS"]
    else:
        weights = args.weights

    if args.filetypes is None:
        filetypes = config.CONFIG["FILETYPES"]["VID"]
    else:
        filetypes = args.filetypes

    if args.overwrite is None:
        overwrite = config.CONFIG["DETECT"]["OVERWRITE"]
    else:
        overwrite = args.overwrite

    if args.debug is None:
        debug = config.CONFIG["DETECT"]["DEBUG"]
    else:
        debug = args.debug

    log.info("Starting detection from command line")
    log.info(f"Arguments: {vars(args)}")

    OTVision.detect(
        paths=paths,
        weights=weights,
        filetypes=filetypes,
        overwrite=overwrite,
        debug=debug,
    )
    log.info("Finished detection from command line")


if __name__ == "__main__":
    main()
