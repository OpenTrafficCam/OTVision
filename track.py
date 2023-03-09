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
import logging
from pathlib import Path

import OTVision.config as config
from OTVision.helpers.log import LOGGER_NAME, VALID_LOG_LEVELS, log
from OTVision.track.track import main as track


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
    parser.add_argument(
        "--log_level_console",
        type=str,
        choices=VALID_LOG_LEVELS,
        help="Log level for logging to the console",
        required=False,
    )
    parser.add_argument(
        "--log_level_file",
        type=str,
        choices=VALID_LOG_LEVELS,
        help="Log level for logging to a log file",
        required=False,
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        help="Path to directory to write the log files",
        required=False,
    )
    return parser.parse_args()


def _process_config(args: argparse.Namespace) -> None:
    if args.config:
        config.parse_user_config(args.config)
    else:
        user_config_cwd = Path(__file__).parent / "user_config.otvision.yaml"

        if user_config_cwd.exists():
            config.parse_user_config(str(user_config_cwd))


def _process_parameters(
    args: argparse.Namespace, log: logging.Logger
) -> tuple[list[Path], float, float, float, int, int, bool]:
    try:
        str_paths = _extract_paths(args)
    except IOError:
        log.exception(
            f"Unable to extract pathlib.Path from the paths you specified: {str_paths}"
        )
        raise
    except Exception:
        log.exception("")
        raise

    paths = [Path(str_path) for str_path in str_paths]

    if args.sigma_l is None:
        sigma_l = config.CONFIG[config.TRACK][config.IOU][config.SIGMA_L]
    else:
        sigma_l = args.sigma_l

    if args.sigma_h is None:
        sigma_h = config.CONFIG[config.TRACK][config.IOU][config.SIGMA_H]
    else:
        sigma_h = args.sigma_h

    if args.sigma_iou is None:
        sigma_iou = config.CONFIG[config.TRACK][config.IOU][config.SIGMA_IOU]
    else:
        sigma_iou = args.sigma_iou

    if args.t_min is None:
        t_min = config.CONFIG[config.TRACK][config.IOU][config.T_MIN]
    else:
        t_min = args.t_min

    if args.t_miss_max is None:
        t_miss_max = config.CONFIG[config.TRACK][config.IOU][config.T_MISS_MAX]
    else:
        t_miss_max = args.t_miss_max

    if args.overwrite is None:
        overwrite = config.CONFIG[config.TRACK][config.OVERWRITE]
    else:
        overwrite = args.overwrite

    return paths, sigma_l, sigma_h, sigma_iou, t_min, t_miss_max, overwrite


def _extract_paths(args: argparse.Namespace) -> list[str]:
    if args.paths:
        return args.paths
    if len(config.CONFIG[config.TRACK][config.PATHS]) == 0:
        raise IOError(
            "No paths have been passed as command line args."
            "No paths have been defined in the user config."
        )

    return config.CONFIG[config.TRACK][config.PATHS]


# TODO: Refactor/outsource this function, as it is redundant in each CLI script
def _configure_logger(args: argparse.Namespace) -> logging.Logger:
    if args.log_level_console is None:
        log_level_console = config.CONFIG[config.LOG][config.LOG_LEVEL_CONSOLE]
    else:
        log_level_console = args.log_level_console

    if args.log_level_file is None:
        log_level_file = config.CONFIG[config.LOG][config.LOG_LEVEL_FILE]
    else:
        log_level_file = args.log_level_file

    if args.log_dir is None:
        try:
            log_dir = Path(config.CONFIG[config.LOG][config.LOG_DIR])
        except TypeError:
            print("No valid LOG_DIR specified in config, check your config file")
            raise
    else:
        log_dir = Path(args.log_dir)

    log.add_console_handler(level=log_level_console)

    log.add_file_handler(log_dir=log_dir, level=log_level_file)

    return logging.getLogger(LOGGER_NAME)


def main() -> None:  # sourcery skip: assign-if-exp
    args = parse()

    _process_config(args)

    log = _configure_logger(args)

    (
        paths,
        sigma_l,
        sigma_h,
        sigma_iou,
        t_min,
        t_miss_max,
        overwrite,
    ) = _process_parameters(args, log)

    log.info("Call track from command line")
    log.info(f"Arguments: {vars(args)}")

    try:
        track(
            paths=paths,
            sigma_l=sigma_l,
            sigma_h=sigma_h,
            sigma_iou=sigma_iou,
            t_min=t_min,
            t_miss_max=t_miss_max,
            overwrite=overwrite,
        )
    except FileNotFoundError:
        log.exception(f"One of the following files cannot be found: {paths}")
        raise
    except Exception:
        log.exception("")
        raise


if __name__ == "__main__":
    main()
