"""
OTVision script to call the track main with arguments parsed from command line
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
    parser = argparse.ArgumentParser(description="Track objects through detections")
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        type=str,
        help="Path or list of paths to detections files",
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
    parser.add_argument(
        "--sigma_l",
        type=float,
        help="Set sigma_l paramter for tracking",
    )
    parser.add_argument(
        "--sigma_h",
        type=float,
        help="Set sigma_h paramter for tracking",
    )
    parser.add_argument(
        "--sigma_iou",
        type=float,
        help="Set sigma_iou paramter for tracking",
    )
    parser.add_argument(
        "--t_min",
        type=int,
        help="Set t_min paramter for tracking",
    )
    parser.add_argument(
        "--t_miss_max",
        type=int,
        help="Set t_miss_max paramter for tracking",
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
        if len(config.CONFIG["TRACK"]["PATHS"]) == 0:
            raise IOError(
                "No paths have been passed as command line args."
                "No paths have been defined in the user config."
            )

        str_paths = config.CONFIG["TRACK"]["PATHS"]
    return str_paths


def main() -> None:
    args = parse()
    _process_config(args)
    try:
        str_paths = _extract_paths(args)
    except IOError as ioe:
        log.error(ioe)

    paths = [Path(str_path) for str_path in str_paths]
    overwrite = args.overwrite or config.CONFIG["TRACK"]["OVERWRITE"]
    debug = args.debug or config.CONFIG["TRACK"]["DEBUG"]
    sigma_l = args.sigma_l or CONFIG["TRACK"]["IOU"]["SIGMA_L"]
    sigma_h = args.sigma_h or CONFIG["TRACK"]["IOU"]["SIGMA_H"]
    sigma_iou = args.sigma_iou or CONFIG["TRACK"]["IOU"]["SIGMA_IOU"]
    t_min = args.t_min or CONFIG["TRACK"]["IOU"]["T_MIN"]
    t_miss_max = args.t_miss_max or CONFIG["TRACK"]["IOU"]["T_MISS_MAX"]

    log.info("Starting tracking from command line")
    log.info(f"Arguments: {vars(args)}")

    track(
        paths=paths,
        sigma_l=sigma_l,
        sigma_h=sigma_h,
        sigma_iou=sigma_iou,
        t_min=t_min,
        t_miss_max=t_miss_max,
        overwrite=overwrite,
        debug=debug,
    )

    log.info("Finished tracking from command line")


if __name__ == "__main__":
    main()
