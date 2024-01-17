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
import logging
from datetime import timedelta
from pathlib import Path

import OTVision
import OTVision.config as config
from OTVision.detect.yolo import loadmodel
from OTVision.helpers.files import check_if_all_paths_exist
from OTVision.helpers.log import DEFAULT_LOG_FILE, LOGGER_NAME, VALID_LOG_LEVELS, log


class ParseError(Exception):
    pass


def parse(argv: list[str] | None) -> argparse.Namespace:
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
        "-w",
        "--weights",
        type=str,
        help="Name of weights from PyTorch hub or Path to weights file",
        required=False,
    )
    parser.add_argument(
        "--conf",
        type=float,
        help="The YOLOv5 models confidence threshold.",
        required=False,
    )
    parser.add_argument(
        "--iou",
        type=float,
        help="The YOLOv5 models IOU threshold.",
        required=False,
    )
    parser.add_argument(
        "--imagesize",
        type=int,
        help="YOLOv5 image size.",
        required=False,
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        help="Use half precision for detection.",
    )
    parser.add_argument(
        "--expected_duration",
        type=int,
        help="Expected duration of a single video in seconds.",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing output files",
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
        "--logfile",
        default=str(DEFAULT_LOG_FILE),
        type=str,
        help="Specify log file directory.",
        required=False,
    )
    parser.add_argument(
        "--logfile_overwrite",
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
) -> tuple[list[Path], str, float, float, int, timedelta, bool, bool]:
    try:
        paths = _extract_paths(args)
    except IOError:
        log.exception("Unable to extract paths from command line or config.yaml")
        raise
    except Exception:
        log.exception("")
        raise

    if args.weights is None:
        weights = config.CONFIG[config.DETECT][config.YOLO][config.WEIGHTS]
    else:
        weights = args.weights

    if args.conf is None:
        conf = config.CONFIG[config.DETECT][config.YOLO][config.CONF]
    else:
        conf = args.conf

    if args.iou is None:
        iou = config.CONFIG[config.DETECT][config.YOLO][config.IOU]
    else:
        iou = args.iou

    if args.imagesize is None:
        imagesize = config.CONFIG[config.DETECT][config.YOLO][config.IMG_SIZE]
    else:
        imagesize = args.imagesize

    # TODO: Future Work: instead of checking each CLI option for existence and
    # returning them, overwrite all passed CLI options in the config object.
    # Get rid of the config.CONFIG dictionary entirely and pass the updated
    # Config object to the detect function. Required options should then be
    # declared in the Config class itself. Their absence must raise a ParseException.
    if args.expected_duration is None:
        config_expected_duration: int | None = config.CONFIG[config.DETECT][
            config.EXPECTED_DURATION
        ]
        if config_expected_duration is None:
            raise ParseError(
                "Required option 'expected duration' is missing! "
                "It must be specified in the config yaml file under "
                "key 'EXPECTED_DURATION' "
                "or passed as CLI option 'expected_duration'."
            )
        expected_duration = timedelta(seconds=config_expected_duration)
    else:
        expected_duration = timedelta(seconds=args.expected_duration)

    if args.half is None:
        half = config.CONFIG[config.DETECT][config.HALF_PRECISION]
    else:
        half = args.half

    if args.overwrite is None:
        overwrite = config.CONFIG[config.DETECT][config.OVERWRITE]
    else:
        overwrite = args.overwrite
    return (
        paths,
        weights,
        conf,
        iou,
        imagesize,
        expected_duration,
        half,
        overwrite,
    )


def _extract_paths(args: argparse.Namespace) -> list[Path]:
    if args.paths is None:
        str_paths = config.CONFIG[config.DETECT][config.PATHS]
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
        weights,
        conf,
        iou,
        imagesize,
        expected_duration,
        half,
        overwrite,
    ) = _process_parameters(args, log)

    log.info("Call detect from command line")
    log.info(f"Arguments: {vars(args)}")

    model = loadmodel(
        weights=weights,
        confidence=conf,
        iou=iou,
        img_size=imagesize,
        half_precision=half,
        normalized=config.CONFIG[config.DETECT][config.YOLO][config.NORMALIZED],
    )

    try:
        OTVision.detect(
            model=model,
            paths=paths,
            expected_duration=expected_duration,
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
