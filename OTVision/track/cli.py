from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from pathlib import Path
from typing import Any

from OTVision.domain.cli import CliParseError, TrackCliArgs, TrackCliParser
from OTVision.helpers.files import check_if_all_paths_exist
from OTVision.helpers.log import DEFAULT_LOG_FILE, VALID_LOG_LEVELS


class ArgparseTrackCliParser(TrackCliParser):
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
            help="Path or list of paths to detections files",
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
            "-o",
            "--overwrite",
            action=BooleanOptionalAction,
            help="Overwrite existing output files",
        )
        self._parser.add_argument(
            "--sigma-l",
            type=float,
            help="Set sigma_l parameter for tracking",
        )
        self._parser.add_argument(
            "--sigma-h",
            type=float,
            help="Set sigma_h parameter for tracking",
        )
        self._parser.add_argument(
            "--sigma-iou",
            type=float,
            help="Set sigma_iou parameter for tracking",
        )
        self._parser.add_argument(
            "--t-min",
            type=int,
            help="Set t_min parameter for tracking",
        )
        self._parser.add_argument(
            "--t-miss-max",
            type=int,
            help="Set t_miss_max parameter for tracking",
        )
        self._parser.add_argument(
            "--tracker",
            type=str,
            help=(
                "Tracker type: 'iou' (default), 'bytetrack', 'botsort', "
                "'ocsort', 'strongsort', 'deepocsort', 'boosttrack', 'hybridsort'. "
                "Non-iou values enable BOXMOT tracking."
            ),
        )
        self._parser.add_argument(
            "--tracker-device",
            type=str,
            help="Device for tracker (e.g., 'cpu', 'cuda:0')",
        )
        self._parser.add_argument(
            "--tracker-half-precision",
            action=BooleanOptionalAction,
            help="Enable FP16 half precision for tracker (GPU only)",
        )
        self._parser.add_argument(
            "--tracker-reid-weights",
            type=str,
            help="Path to ReID model weights for appearance-based trackers",
        )
        self._parser.add_argument(
            "--tracker-params",
            nargs="*",
            type=str,
            help=(
                "Tracker parameters as key=value pairs "
                "(e.g., --tracker-params track_buffer=60 match_thresh=0.8)"
            ),
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

    def parse(self) -> TrackCliArgs:
        args = self._parser.parse_args(self._argv)
        self.__assert_cli_args_valid(args)

        return TrackCliArgs(
            paths=self._parse_files(args.paths),
            config_file=args.config,
            overwrite=args.overwrite,
            sigma_l=float(args.sigma_l) if args.sigma_l is not None else None,
            sigma_h=float(args.sigma_h) if args.sigma_h is not None else None,
            sigma_iou=float(args.sigma_iou) if args.sigma_iou is not None else None,
            t_min=int(args.t_min) if args.t_min is not None else None,
            t_miss_max=int(args.t_miss_max) if args.t_miss_max is not None else None,
            tracker=args.tracker,
            tracker_device=args.tracker_device,
            tracker_half_precision=args.tracker_half_precision,
            tracker_reid_weights=args.tracker_reid_weights,
            tracker_params=self._parse_tracker_params(args.tracker_params),
            logfile=Path(args.logfile),
            log_level_console=args.log_level_console,
            log_level_file=args.log_level_file,
            logfile_overwrite=args.logfile_overwrite,
        )

    def _parse_tracker_params(self, params: list[str] | None) -> dict[str, Any] | None:
        """Parse tracker parameters from key=value format.

        Args:
            params: List of strings in key=value format

        Returns:
            Dictionary of parsed parameters, or None if no params provided

        Raises:
            CliParseError: If a parameter is not in key=value format
        """
        if params is None:
            return None

        result: dict[str, Any] = {}
        for param in params:
            if "=" not in param:
                raise CliParseError(
                    f"Invalid tracker parameter format: '{param}'. Expected: key=value"
                )
            key, value = param.split("=", 1)
            # Parse as int, then float, else string
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
        return result if result else None

    def __assert_cli_args_valid(self, args: Namespace) -> None:
        if args.paths is None and args.config is None:
            raise CliParseError(
                (
                    "No paths have been passed as command line args."
                    "No user config has been passed as command line arg."
                )
            )

    def _parse_files(self, files: list[str] | None) -> list[str] | None:
        if files is None:
            return None

        result = [Path(file).expanduser() for file in files]
        check_if_all_paths_exist(result)
        return list(map(str, result))
