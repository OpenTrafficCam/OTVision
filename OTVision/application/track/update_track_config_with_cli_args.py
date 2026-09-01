import logging

from OTVision.abstraction.defaults import value_or_default
from OTVision.application.config import Config, TrackConfig, _LogConfig, _TrackIouConfig
from OTVision.application.track.get_track_cli_args import GetTrackCliArgs
from OTVision.domain.cli import TrackCliArgs
from OTVision.domain.tracker import TrackerType
from OTVision.helpers.log import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)

IGNORED_IOU_CLI_ARGS = ("sigma_l", "sigma_h", "sigma_iou", "t_min", "t_miss_max")


class UpdateTrackConfigWithCliArgs:
    def __init__(self, get_detect_cli_args: GetTrackCliArgs) -> None:
        self._get_track_cli_args = get_detect_cli_args

    def update(self, config: Config) -> Config:
        cli_args = self._get_track_cli_args.get()
        return Config(
            log=self._update_log_config(config, cli_args),
            search_subdirs=config.search_subdirs,
            default_filetype=config.default_filetype,
            last_paths=config.last_paths,
            convert=config.convert,
            detect=config.detect,
            track=self._update_track_config(config.track, cli_args),
            undistort=config.undistort,
            transform=config.transform,
            gui=config.gui,
            stream=config.stream,
        )

    def _warn_about_ignored_iou_args(self, cli_args: TrackCliArgs) -> None:
        """Warn that IOU-only CLI overrides are ignored in BoT-SORT mode.

        Args:
            cli_args (TrackCliArgs): Parsed track CLI arguments.
        """
        ignored = [
            name for name in IGNORED_IOU_CLI_ARGS if getattr(cli_args, name) is not None
        ]
        if ignored:
            log.warning(
                "botsort mode ignores the IOU CLI override(s) %s; "
                "configure BoT-SORT under TRACK.BOT_SORT in YAML instead.",
                ", ".join(sorted(ignored)),
            )

    def _update_track_config(
        self, track_config: TrackConfig, cli_args: TrackCliArgs
    ) -> TrackConfig:
        resolved_tracker_type = value_or_default(
            cli_args.tracker_type, track_config.tracker_type
        )
        if resolved_tracker_type is TrackerType.BOTSORT:
            self._warn_about_ignored_iou_args(cli_args)
            iou_config = track_config.iou
        else:
            iou = track_config.iou
            iou_config = _TrackIouConfig(
                sigma_l=value_or_default(cli_args.sigma_l, iou.sigma_l),
                sigma_h=value_or_default(cli_args.sigma_h, iou.sigma_h),
                sigma_iou=value_or_default(cli_args.sigma_iou, iou.sigma_iou),
                t_min=value_or_default(cli_args.t_min, iou.t_min),
                t_miss_max=value_or_default(cli_args.t_miss_max, iou.t_miss_max),
            )
        return TrackConfig(
            paths=value_or_default(cli_args.paths, track_config.paths),
            run_chained=track_config.run_chained,
            iou=iou_config,
            tracker_type=resolved_tracker_type,
            botsort=track_config.botsort,
            overwrite=value_or_default(cli_args.overwrite, track_config.overwrite),
        )

    def _update_log_config(self, config: Config, cli_args: TrackCliArgs) -> _LogConfig:
        return _LogConfig(
            log_level_console=value_or_default(
                cli_args.log_level_console, config.log.log_level_console
            ),
            log_level_file=value_or_default(
                cli_args.log_level_file, config.log.log_level_file
            ),
        )
