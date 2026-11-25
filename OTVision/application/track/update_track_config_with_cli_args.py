from OTVision.abstraction.defaults import value_or_default
from OTVision.application.config import (
    TRACKER_TYPE_BOT_SORT,
    Config,
    TrackConfig,
    _LogConfig,
    _TrackIouConfig,
)
from OTVision.application.config_parser import (
    ConfigParser,
    InvalidOtvisionConfigError,
    validate_tracker_type,
)
from OTVision.application.track.get_track_cli_args import GetTrackCliArgs
from OTVision.domain.cli import TrackCliArgs


class UpdateTrackConfigWithCliArgs:
    def __init__(self, get_detect_cli_args: GetTrackCliArgs) -> None:
        self._get_track_cli_args = get_detect_cli_args

    def update(self, config: Config) -> Config:
        cli_args = self._get_track_cli_args.get()
        updated = Config(
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
        ConfigParser.validate_flush_buffer_support_track_lifecycle(updated)
        return updated

    def _update_track_config(
        self, track_config: TrackConfig, cli_args: TrackCliArgs
    ) -> TrackConfig:
        tracker_type = value_or_default(
            cli_args.tracker_type, track_config.tracker_type
        )
        validate_tracker_type(tracker_type)
        if tracker_type == TRACKER_TYPE_BOT_SORT:
            iou_options = (
                ("--sigma-l", cli_args.sigma_l),
                ("--sigma-h", cli_args.sigma_h),
                ("--sigma-iou", cli_args.sigma_iou),
                ("--t-min", cli_args.t_min),
                ("--t-miss-max", cli_args.t_miss_max),
            )
            for option, value in iou_options:
                if value is not None:
                    raise InvalidOtvisionConfigError(
                        f"{option} is an IOU option and cannot be used with BoT-SORT."
                    )
            ConfigParser.validate_track_bot_sort_config(track_config.bot_sort)

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
            overwrite=value_or_default(cli_args.overwrite, track_config.overwrite),
            bot_sort=track_config.bot_sort,
            tracker_type=tracker_type,
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
