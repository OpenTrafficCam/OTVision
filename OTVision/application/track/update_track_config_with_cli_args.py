from OTVision.abstraction.defaults import value_or_default
from OTVision.application.config import (
    Config,
    TrackConfig,
    _LogConfig,
    _TrackBoxmotConfig,
    _TrackIouConfig,
)
from OTVision.application.track.get_track_cli_args import GetTrackCliArgs
from OTVision.domain.cli import TrackCliArgs


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

    def _update_track_config(
        self, track_config: TrackConfig, cli_args: TrackCliArgs
    ) -> TrackConfig:
        iou_config = _TrackIouConfig(
            sigma_l=value_or_default(cli_args.sigma_l, track_config.sigma_l),
            sigma_h=value_or_default(cli_args.sigma_h, track_config.sigma_h),
            sigma_iou=value_or_default(cli_args.sigma_iou, track_config.sigma_iou),
            t_min=value_or_default(cli_args.t_min, track_config.t_min),
            t_miss_max=value_or_default(cli_args.t_miss_max, track_config.t_miss_max),
        )
        boxmot_config = self._update_boxmot_config(track_config.boxmot, cli_args)
        return TrackConfig(
            paths=value_or_default(cli_args.paths, track_config.paths),
            run_chained=track_config.run_chained,
            iou=iou_config,
            boxmot=boxmot_config,
            overwrite=value_or_default(cli_args.overwrite, track_config.overwrite),
        )

    def _update_boxmot_config(
        self, boxmot_config: _TrackBoxmotConfig, cli_args: TrackCliArgs
    ) -> _TrackBoxmotConfig:
        """Update BOXMOT config with CLI arguments.

        If --tracker is specified:
        - 'iou' disables BOXMOT
        - Any other value enables BOXMOT with that tracker type

        Args:
            boxmot_config: The existing BOXMOT configuration
            cli_args: CLI arguments to apply

        Returns:
            Updated BOXMOT configuration
        """
        if cli_args.tracker is not None:
            enabled = cli_args.tracker.lower() != "iou"
            tracker_type = cli_args.tracker
        else:
            enabled = boxmot_config.enabled
            tracker_type = boxmot_config.tracker_type

        return _TrackBoxmotConfig(
            enabled=enabled,
            tracker_type=tracker_type,
            device=value_or_default(cli_args.tracker_device, boxmot_config.device),
            half_precision=value_or_default(
                cli_args.tracker_half_precision, boxmot_config.half_precision
            ),
            reid_weights=value_or_default(
                cli_args.tracker_reid_weights, boxmot_config.reid_weights
            ),
            tracker_params=value_or_default(
                cli_args.tracker_params, boxmot_config.tracker_params
            ),
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
