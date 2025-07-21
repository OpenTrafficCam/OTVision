from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from pathlib import Path

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
            "--botsort-config",
            type=str,
            help="Path to BoT-SORT configuration yaml file.",
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
            "--tracker-type",
            type=str,
            choices=["iou", "botsort"],
            help="Select tracking algorithm: 'iou' or 'botsort'",
        )
        # BoT-SORT parameters
        self._parser.add_argument(
            "--track-high-thresh",
            type=float,
            help="Set track_high_thresh parameter for BoT-SORT tracking",
        )
        self._parser.add_argument(
            "--track-low-thresh",
            type=float,
            help="Set track_low_thresh parameter for BoT-SORT tracking",
        )
        self._parser.add_argument(
            "--new-track-thresh",
            type=float,
            help="Set new_track_thresh parameter for BoT-SORT tracking",
        )
        self._parser.add_argument(
            "--track-buffer",
            type=int,
            help="Set track_buffer parameter for BoT-SORT tracking",
        )
        self._parser.add_argument(
            "--match-thresh",
            type=float,
            help="Set match_thresh parameter for BoT-SORT tracking",
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
            botsort_config_file=(
                Path(args.botsort_config) if args.botsort_config else None
            ),
            overwrite=args.overwrite,
            tracker_type=args.tracker_type,
            sigma_l=float(args.sigma_l) if args.sigma_l is not None else None,
            sigma_h=float(args.sigma_h) if args.sigma_h is not None else None,
            sigma_iou=float(args.sigma_iou) if args.sigma_iou is not None else None,
            t_min=int(args.t_min) if args.t_min is not None else None,
            t_miss_max=int(args.t_miss_max) if args.t_miss_max is not None else None,
            track_high_thresh=(
                float(args.track_high_thresh)
                if args.track_high_thresh is not None
                else None
            ),
            track_low_thresh=(
                float(args.track_low_thresh)
                if args.track_low_thresh is not None
                else None
            ),
            new_track_thresh=(
                float(args.new_track_thresh)
                if args.new_track_thresh is not None
                else None
            ),
            track_buffer=(
                int(args.track_buffer) if args.track_buffer is not None else None
            ),
            match_thresh=(
                float(args.match_thresh) if args.match_thresh is not None else None
            ),
            logfile=Path(args.logfile),
            log_level_console=args.log_level_console,
            log_level_file=args.log_level_file,
            logfile_overwrite=args.logfile_overwrite,
        )

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
