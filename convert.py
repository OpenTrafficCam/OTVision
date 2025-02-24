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
import logging
from pathlib import Path

import OTVision
import OTVision.config as config
from OTVision.helpers.files import check_if_all_paths_exist
from OTVision.helpers.log import DEFAULT_LOG_FILE, LOGGER_NAME, VALID_LOG_LEVELS, log


def parse(argv: list[str] | None) -> argparse.Namespace:
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
        "--delete-input",
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
        "--input-fps",
        type=float,
        help="Frame rate of input h264.",
        required=False,
    )
    parser.add_argument(
        "-r",
        "--rotation",
        type=int,
        help="Add rotation information to video metadata.",
        required=False,
    )
    parser.add_argument(
        "--fps-from-filename",
        action=argparse.BooleanOptionalAction,
        help="Whether or not to parse frame rate from file name.",
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
) -> tuple[list[Path], float, bool, int, bool, bool]:
    try:
        paths = _extract_paths(args)
    except IOError:
        log.exception("Unable to extract paths from command line or config.yaml")
        raise
    except Exception:
        log.exception("")
        raise

    if args.input_fps is None:
        input_fps = config.CONFIG[config.CONVERT][config.INPUT_FPS]
    else:
        input_fps = args.input_fps

    if args.fps_from_filename is None:
        fps_from_filename = config.CONFIG[config.CONVERT][config.FPS_FROM_FILENAME]
    else:
        fps_from_filename = args.fps_from_filename

    if args.rotation is None:
        rotation = config.CONFIG[config.CONVERT][config.ROTATION]
    else:
        rotation = args.rotation

    if args.overwrite is None:
        overwrite = config.CONFIG[config.CONVERT][config.OVERWRITE]
    else:
        overwrite = args.overwrite

    if args.delete_input is None:
        delete_input = config.CONFIG[config.CONVERT][config.DELETE_INPUT]
    else:
        delete_input = args.delete_input
    return paths, input_fps, fps_from_filename, rotation, overwrite, delete_input


def _extract_paths(args: argparse.Namespace) -> list[Path]:
    if args.paths is None:
        str_paths = config.CONFIG[config.CONVERT][config.PATHS]
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


def main(argv: list[str] | None = None) -> None:
    args = parse(argv)

    _process_config(args)

    log = _configure_logger(args)

    (
        paths,
        input_fps,
        fps_from_filename,
        rotation,
        overwrite,
        delete_input,
    ) = _process_parameters(args, log)

    log.info("Call convert from command line")
    log.info(f"Arguments: {vars(args)}")

    try:
        OTVision.convert(
            paths=paths,
            input_fps=input_fps,
            fps_from_filename=fps_from_filename,
            rotation=rotation,
            overwrite=overwrite,
            delete_input=delete_input,
        )
    except FileNotFoundError:
        log.exception(f"One of the following files cannot be found: {paths}")
        raise
    except Exception:
        log.exception("")
        raise


if __name__ == "__main__":
    main()
