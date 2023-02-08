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

import OTVision
import OTVision.config as config
from OTVision.helpers.log import log


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert h264 to mp4")
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        type=str,
        help="Path or list of paths to h264 (or other) video files",
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
        "--delete_input",
        action=argparse.BooleanOptionalAction,
        help="Delete input files after convert",
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
        str_paths = args.paths
    else:
        if len(config.CONFIG["CONVERT"]["PATHS"]) == 0:
            raise IOError(
                "No paths have been passed as command line args.\n"
                "No paths have been defined in the user config."
            )

        str_paths = config.CONFIG["CONVERT"]["PATHS"]
    return str_paths


def main() -> None:
    args = parse()
    _process_config(args)
    try:
        str_paths = _extract_paths(args)
    except IOError as ioe:
        log.error(ioe)

    delete_input = args.delete_input or config.CONFIG["CONVERT"]["DELETE_INPUT"]
    overwrite = args.overwrite or config.CONFIG["CONVERT"]["OVERWRITE"]
    debug = args.debug or config.CONFIG["CONVERT"]["DEBUG"]
    paths = [Path(str_path) for str_path in str_paths]

    log.info("Starting conversion from command line")
    log.info(f"Arguments: {vars(args)}")

    OTVision.convert(
        paths=paths,
        delete_input=delete_input,
        overwrite=overwrite,
        debug=debug,
    )
    log.info("Finished conversion from command line")


if __name__ == "__main__":
    main()
