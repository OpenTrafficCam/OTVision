from OTVision.application.detect.get_detect_cli_args import GetDetectCliArgs
from OTVision.config import Config, DetectConfig, YoloConfig, _LogConfig
from OTVision.domain.cli import DetectCliArgs


class UpdateDetectConfigWithCliArgs:
    def __init__(self, get_detect_cli_args: GetDetectCliArgs) -> None:
        self._get_detect_cli_args = get_detect_cli_args

    def update(self, config: Config) -> Config:
        cli_args = self._get_detect_cli_args.get()
        return Config(
            log=self._update_log_config(config, cli_args),
            search_subdirs=config.search_subdirs,
            default_filetype=config.default_filetype,
            last_paths=config.last_paths,
            convert=config.convert,
            detect=self._update_detect_config(config.detect, cli_args),
            track=config.track,
            undistort=config.undistort,
            transform=config.transform,
            gui=config.gui,
        )

    def _update_detect_config(
        self, detect_config: DetectConfig, cli_args: DetectCliArgs
    ) -> DetectConfig:
        yolo_config = YoloConfig(
            weights=(
                cli_args.weights
                if cli_args.weights is not None
                else detect_config.yolo_config.weights
            ),
            available_weights=detect_config.yolo_config.available_weights,
            conf=(
                cli_args.conf
                if cli_args.conf is not None
                else detect_config.yolo_config.conf
            ),
            iou=(
                cli_args.iou
                if cli_args.iou is not None
                else detect_config.yolo_config.iou
            ),
            img_size=(
                cli_args.imagesize
                if cli_args.imagesize is not None
                else detect_config.yolo_config.img_size
            ),
            chunk_size=detect_config.yolo_config.chunk_size,
            normalized=detect_config.yolo_config.normalized,
        )
        return DetectConfig(
            paths=cli_args.paths if cli_args.paths is not None else detect_config.paths,
            yolo_config=yolo_config,
            expected_duration=(
                cli_args.expected_duration
                if cli_args.expected_duration is not None
                else detect_config.expected_duration
            ),
            overwrite=(
                cli_args.overwrite
                if cli_args.overwrite is not None
                else detect_config.overwrite
            ),
            half_precision=(
                cli_args.half
                if cli_args.half is not None
                else detect_config.half_precision
            ),
            start_time=(
                cli_args.start_time
                if cli_args.start_time is not None
                else detect_config.start_time
            ),
            detect_start=(
                cli_args.detect_start
                if cli_args.detect_start is not None
                else detect_config.detect_start
            ),
            detect_end=(
                cli_args.detect_end
                if cli_args.detect_end is not None
                else detect_config.detect_end
            ),
        )

    def _update_log_config(self, config: Config, cli_args: DetectCliArgs) -> _LogConfig:
        return _LogConfig(
            log_level_console=(
                cli_args.log_level_console
                if cli_args.log_level_console is not None
                else config.log.log_level_console
            ),
            log_level_file=(
                cli_args.log_level_file
                if cli_args.log_level_file is not None
                else config.log.log_level_file
            ),
        )
