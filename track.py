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

from OTVision.config import CONFIG
from OTVision.helpers.log import log
from OTVision.track.track import main as track


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track objects through detections")
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        type=str,
        help="Path or list of paths to detections files",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        default=CONFIG["TRACK"]["OVERWRITE"],
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "-d",
        "--debug",
        default=CONFIG["TRACK"]["DEBUG"],
        action=argparse.BooleanOptionalAction,
        help="Logging in debug mode",
    )
    parser.add_argument(
        "--sigma_l",
        default=CONFIG["TRACK"]["IOU"]["SIGMA_L"],
        type=float,
        help="Set sigma_l paramter for tracking",
    )
    parser.add_argument(
        "--sigma_h",
        default=CONFIG["TRACK"]["IOU"]["SIGMA_H"],
        type=float,
        help="Set sigma_h paramter for tracking",
    )
    parser.add_argument(
        "--sigma_iou",
        default=CONFIG["TRACK"]["IOU"]["SIGMA_IOU"],
        type=float,
        help="Set sigma_iou paramter for tracking",
    )
    parser.add_argument(
        "--t_min",
        default=CONFIG["TRACK"]["IOU"]["T_MIN"],
        type=int,
        help="Set t_min paramter for tracking",
    )
    parser.add_argument(
        "--t_miss_max",
        default=CONFIG["TRACK"]["IOU"]["T_MISS_MAX"],
        type=int,
        help="Set t_miss_max paramter for tracking",
    )
    return parser.parse_args()


def main() -> None:
    args = parse()
    paths = [Path(str_path) for str_path in args.paths]
    log.info("Starting tracking from command line")
    log.info(f"Arguments: {vars(args)}")
    track(
        paths=paths,
        overwrite=args.overwrite,
        debug=args.debug,
        sigma_l=args.sigma_l,
        sigma_h=args.sigma_h,
        sigma_iou=args.sigma_iou,
        t_min=args.t_min,
        t_miss_max=args.t_miss_max,
    )
    log.info("Finished tracking from command line")


if __name__ == "__main__":
    main()
