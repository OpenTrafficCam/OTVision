from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from datetime import datetime, timedelta
from pathlib import Path

from OTVision.config import DATETIME_FORMAT
from OTVision.domain.cli import CliParseError, DetectCliArgs, DetectCliParser
from OTVision.helpers.files import check_if_all_paths_exist
from OTVision.helpers.log import DEFAULT_LOG_FILE, VALID_LOG_LEVELS


class ArgparseDetectCliParser(DetectCliParser):
    def __init__(
        self,
        parser: ArgumentParser,
        argv: list[str] | None = None,
    ) -> None:
        self._parser = parser
        self._argv = argv
        self.__setup()

    def __setup(self) -> None:
        self._parser.add_argument(
            "-p",
            "--paths",
            nargs="+",
            type=str,
            help=(
                "Path/list of paths to image or video or folder "
                "containing videos/images"
            ),
            required=False,
        )
        self._parser.add_argument(
            "-c",
            "--config",
            type=str,
            help="Path to custom user configuration yaml file.",
            required=False,
        )
        self._parser.add_argument(
            "-w",
            "--weights",
            type=str,
            help="Name of weights from PyTorch hub or Path to weights file",
            required=False,
        )
        self._parser.add_argument(
            "--conf",
            type=float,
            help="The YOLOv5 models confidence threshold.",
            required=False,
        )
        self._parser.add_argument(
            "--iou",
            type=float,
            help="The YOLOv5 models IOU threshold.",
            required=False,
        )
        self._parser.add_argument(
            "--imagesize",
            type=int,
            help="YOLOv5 image size.",
            required=False,
        )
        self._parser.add_argument(
            "--half",
            action=BooleanOptionalAction,
            help="Use half precision for detection.",
        )
        self._parser.add_argument(
            "--expected-duration",
            type=int,
            help="Expected duration of a single video in seconds.",
            required=False,
        )
        self._parser.add_argument(
            "-o",
            "--overwrite",
            action=BooleanOptionalAction,
            help="Overwrite existing output files",
        )
        self._parser.add_argument(
            "--log-level-console",
            type=str,
            choices=VALID_LOG_LEVELS,
            help="Log level for logging to the console",
            required=False,
        )
        self._parser.add_argument(
            "--log-level-file",
            type=str,
            choices=VALID_LOG_LEVELS,
            help="Log level for logging to a log file",
            required=False,
        )
        self._parser.add_argument(
            "--logfile",
            default=str(DEFAULT_LOG_FILE),
            type=str,
            help="Specify log file directory.",
            required=False,
        )
        self._parser.add_argument(
            "--logfile-overwrite",
            action="store_true",
            help="Overwrite log file if it already exists.",
            required=False,
        )
        self._parser.add_argument(
            "--start-time",
            default=None,
            type=str,
            help="Specify start date and time of the recording in format "
            "YYYY-MM-DD_HH-MM-SS",
            required=False,
        )
        self._parser.add_argument(
            "--detect-start",
            default=None,
            type=int,
            help="Specify start of detection in seconds.",
            required=False,
        )
        self._parser.add_argument(
            "--detect-end",
            default=None,
            type=int,
            help="Specify end of detection in seconds.",
            required=False,
        )

    def parse(self) -> DetectCliArgs:
        args = self._parser.parse_args(self._argv)
        self.__assert_cli_args_valid(args)

        return DetectCliArgs(
            paths=self._parse_files(args.paths),
            config_file=args.config,
            weights=args.weights,
            conf=float(args.conf) if args.conf is not None else None,
            iou=float(args.iou) if args.iou is not None else None,
            imagesize=int(args.imagesize) if args.imagesize is not None else None,
            expected_duration=(
                timedelta(seconds=args.expected_duration)
                if args.expected_duration is not None
                else None
            ),
            half=bool(args.half) if args.half else None,
            overwrite=args.overwrite,
            start_time=self._parse_start_time(args.start_time),
            detect_start=(
                int(args.detect_start) if args.detect_start is not None else None
            ),
            detect_end=int(args.detect_end) if args.detect_end is not None else None,
            logfile=Path(args.logfile),
            log_level_console=args.log_level_console,
            log_level_file=args.log_level_file,
            logfile_overwrite=args.logfile_overwrite,
        )

    def _parse_start_time(self, start_time: str | None) -> datetime | None:
        if start_time is None:
            return None
        return datetime.strptime(start_time, DATETIME_FORMAT)

    def __assert_cli_args_valid(self, args: Namespace) -> None:
        if args.paths is None and args.config is None:
            raise CliParseError(
                (
                    "No paths have been passed as command line args."
                    "No paths have been defined in the user config."
                )
            )

    def _parse_files(self, files: list[str] | None) -> list[Path] | None:
        if files is None:
            return None

        result = [Path(file).expanduser() for file in files]
        check_if_all_paths_exist(result)
        return result
