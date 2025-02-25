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

import OTVision
import OTVision.config as config
from OTVision.helpers.files import check_if_all_paths_exist
from OTVision.helpers.log import DEFAULT_LOG_FILE, LOGGER_NAME, VALID_LOG_LEVELS, log


def parse(argv: list[str] | None) -> argparse.Namespace:
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
        "--sigma-l",
        type=float,
        help="Set sigma_l parameter for tracking",
    )
    parser.add_argument(
        "--sigma-h",
        type=float,
        help="Set sigma_h parameter for tracking",
    )
    parser.add_argument(
        "--sigma-iou",
        type=float,
        help="Set sigma_iou parameter for tracking",
    )
    parser.add_argument(
        "--t-min",
        type=int,
        help="Set t_min parameter for tracking",
    )
    parser.add_argument(
        "--t-miss-max",
        type=int,
        help="Set t_miss_max parameter for tracking",
    )
    parser.add_argument(
        "--log-level-console",
        type=str,
        choices=VALID_LOG_LEVELS,
        help="Log level for logging to the console",
        required=False,
    )
    parser.add_argument(
        "--log-level-file",
        type=str,
        choices=VALID_LOG_LEVELS,
        help="Log level for logging to a log file",
        required=False,
    )
    parser.add_argument(
        "--logfile",
        default=str(DEFAULT_LOG_FILE),
        type=str,
        help="Specify log file directory.",
        required=False,
    )
    parser.add_argument(
        "--logfile-overwrite",
        action="store_true",
        help="Overwrite log file if it already exists.",
        required=False,
    )
    return parser.parse_args(argv)


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
        paths = _extract_paths(args)
    except IOError:
        log.exception("Unable to extract paths from command line or config.yaml")
        raise
    except Exception:
        log.exception("")
        raise

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


def _extract_paths(args: argparse.Namespace) -> list[Path]:
    if args.paths is None:
        str_paths = config.CONFIG[config.TRACK][config.PATHS]
    else:
        str_paths = args.paths
    if len(str_paths) == 0:
        raise IOError(
            "No paths have been passed as command line args."
            "No paths have been defined in the user config."
        )
    paths = [Path(str_path).expanduser() for str_path in str_paths]
    check_if_all_paths_exist(paths)

    return paths


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

    log.add_console_handler(level=log_level_console)
    log.add_file_handler(
        Path(args.logfile).expanduser(),
        log_level_file,
        args.logfile_overwrite,
    )
    return logging.getLogger(LOGGER_NAME)


def main(argv: list[str] | None = None) -> None:  # sourcery skip: assign-if-exp
    args = parse(argv)

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
        OTVision.track(
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
