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
import sys
from pathlib import Path

import OTVision
import OTVision.config as config
from OTVision.helpers.files import check_if_all_paths_exist
from OTVision.helpers.log import LOGGER_NAME, VALID_LOG_LEVELS, log


def parse(argv: list[str]) -> argparse.Namespace:
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
        "--chunksize",
        type=int,
        help="The number of frames of a video to be inferred in one iteration.",
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
        "-f",
        "--force",
        help="Force reload model in torch hub instead of using cache.",
        action=argparse.BooleanOptionalAction,
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
        "--log_dir",
        type=str,
        help="Path to directory to write the log files",
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
) -> tuple[list[Path], str, float, float, int, int, bool, bool, bool]:
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

    if args.chunksize is None:
        chunksize = config.CONFIG[config.DETECT][config.YOLO][config.CHUNK_SIZE]
    else:
        chunksize = args.chunksize

    if args.half is None:
        half = config.CONFIG[config.DETECT][config.HALF_PRECISION]
    else:
        half = args.half

    if args.force is None:
        force_reload = config.CONFIG[config.DETECT][config.FORCE_RELOAD_TORCH_HUB_CACHE]
    else:
        force_reload = args.force

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
        chunksize,
        half,
        force_reload,
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
    paths = [Path(str_path) for str_path in str_paths]
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


def main(argv: list[str]) -> int:  # sourcery skip: assign-if-exp
    args = parse(argv)

    _process_config(args)

    log = _configure_logger(args)

    (
        paths,
        weights,
        conf,
        iou,
        imagesize,
        chunksize,
        half,
        force_reload,
        overwrite,
    ) = _process_parameters(args, log)

    log.info("Call detect from command line")
    log.info(f"Arguments: {vars(args)}")

    try:
        OTVision.detect(
            paths=paths,
            weights=weights,
            conf=conf,
            iou=iou,
            size=imagesize,
            chunksize=chunksize,
            half_precision=half,
            force_reload_torch_hub_cache=force_reload,
            overwrite=overwrite,
        )
    except FileNotFoundError:
        log.exception(f"One of the following files cannot be found: {paths}")
        raise
    except Exception:
        log.exception("")
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
