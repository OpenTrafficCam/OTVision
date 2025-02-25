import argparse
import logging
from pathlib import Path

import OTVision.config as config
from OTVision.helpers.log import DEFAULT_LOG_FILE, LOGGER_NAME, VALID_LOG_LEVELS, log
from OTVision.transform.reference_points_picker import ReferencePointsPicker


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference Points Picker")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the video or image in which reference points are to be clicked",
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
    return parser.parse_args()


def _configure_logger(args: argparse.Namespace) -> logging.Logger:
    # Add console handler to existing logger instance

    if args.log_level_console is None:
        log_level_console = config.CONFIG[config.LOG][config.LOG_LEVEL_CONSOLE]
    else:
        log_level_console = args.log_level_console

    log.add_console_handler(level=log_level_console)

    # Add file handler to existing logger instance

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
    # Return and overwrite log variable pointing to same global object from log.py
    return logging.getLogger(LOGGER_NAME)


def main() -> None:
    args = parse()
    log = _configure_logger(args)
    log.info("Call reference points picker from command line")
    log.info(f"Arguments: {vars(args)}")
    ReferencePointsPicker(file=Path(args.file))


if __name__ == "__main__":
    main()
