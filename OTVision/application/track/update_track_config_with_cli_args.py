from OTVision.abstraction.defaults import value_or_default
from OTVision.application.config import (
    Config,
    TrackConfig,
    _LogConfig,
    _TrackBotSortConfig,
    _TrackIouConfig,
)
from OTVision.application.config_parser import ConfigParser
from OTVision.application.track.get_track_cli_args import GetTrackCliArgs
from OTVision.domain.cli import TrackCliArgs
from OTVision.plugin.yaml_serialization import YamlDeserializer


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

        # Load BoT-SORT config from file if provided, otherwise use existing config
        base_botsort_config = track_config.botsort
        if cli_args.botsort_config_file:
            try:
                deserializer = YamlDeserializer()
                botsort_data = deserializer.deserialize(cli_args.botsort_config_file)
                config_parser = ConfigParser(deserializer)
                base_botsort_config = config_parser.parse_track_botsort_config(
                    botsort_data
                )
            except Exception:
                # If loading fails, fall back to existing config
                base_botsort_config = track_config.botsort

        botsort_config = _TrackBotSortConfig(
            track_high_thresh=value_or_default(
                cli_args.track_high_thresh, base_botsort_config.track_high_thresh
            ),
            track_low_thresh=value_or_default(
                cli_args.track_low_thresh, base_botsort_config.track_low_thresh
            ),
            new_track_thresh=value_or_default(
                cli_args.new_track_thresh, base_botsort_config.new_track_thresh
            ),
            track_buffer=value_or_default(
                cli_args.track_buffer, base_botsort_config.track_buffer
            ),
            match_thresh=value_or_default(
                cli_args.match_thresh, base_botsort_config.match_thresh
            ),
        )

        return TrackConfig(
            paths=value_or_default(cli_args.paths, track_config.paths),
            run_chained=track_config.run_chained,
            tracker_type=value_or_default(
                cli_args.tracker_type, track_config.tracker_type
            ),
            iou=iou_config,
            botsort=botsort_config,
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
