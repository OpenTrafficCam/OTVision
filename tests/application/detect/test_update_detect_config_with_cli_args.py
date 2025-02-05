from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

from OTVision.application.detect.update_detect_config_with_cli_args import (
    UpdateDetectConfigWithCliArgs,
)
from OTVision.config import Config, DetectConfig, YoloConfig, _LogConfig
from OTVision.domain.cli import DetectCliArgs

# Define constants for DetectCliArgs arguments
EXPECTED_DURATION = timedelta(seconds=300)
PATHS = [Path("file1.mp4"), Path("file2.mp4")]
CONFIG_FILE = Path("config.yaml")
LOGFILE = Path("logfile.log")
LOGFILE_OVERWRITE = True
LOG_LEVEL_CONSOLE = "DEBUG"
LOG_LEVEL_FILE = "DEBUG"
WEIGHTS = "path/to/custom.weights.pt"
CONF_THRESHOLD = 0.512
IOU_THRESHOLD = 0.612
IMAGE_SIZE = 1280
HALF_PRECISION = True
OVERWRITE = True
DETECT_START = 300
DETECT_END = 600


class TestUpdateDetectConfigWithCliArgs:
    def test_update(self) -> None:
        """
        #Requirement https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7184
        """  # noqa
        given = create_get_detect_cli_args()
        given_config = default_config()
        target = UpdateDetectConfigWithCliArgs(given)

        actual = target.update(given_config)
        assert actual == expected_config()
        given.get.assert_called_once()


def default_config() -> Config:
    return Config()


def create_get_detect_cli_args() -> Mock:
    get_cli_args = Mock()
    get_cli_args.get.return_value = cli_args()
    return get_cli_args


def cli_args() -> DetectCliArgs:
    return DetectCliArgs(
        expected_duration=EXPECTED_DURATION,
        paths=PATHS,
        config_file=CONFIG_FILE,
        logfile=LOGFILE,
        logfile_overwrite=LOGFILE_OVERWRITE,
        log_level_console=LOG_LEVEL_CONSOLE,
        log_level_file=LOG_LEVEL_FILE,
        weights=WEIGHTS,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imagesize=IMAGE_SIZE,
        half=HALF_PRECISION,
        overwrite=OVERWRITE,
        detect_start=DETECT_START,
        detect_end=DETECT_END,
    )


def expected_config() -> Config:
    config = default_config()
    args = cli_args()
    return Config(
        log=_LogConfig(
            log_level_file=LOG_LEVEL_FILE, log_level_console=LOG_LEVEL_CONSOLE
        ),
        search_subdirs=config.search_subdirs,
        default_filetype=config.default_filetype,
        last_paths=config.last_paths,
        convert=config.convert,
        detect=DetectConfig(
            paths=PATHS,
            yolo_config=YoloConfig(
                weights=WEIGHTS,
                available_weights=config.detect.yolo_config.available_weights,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                img_size=IMAGE_SIZE,
                chunk_size=config.detect.yolo_config.chunk_size,
                normalized=config.detect.yolo_config.normalized,
            ),
            expected_duration=args.expected_duration,
            overwrite=OVERWRITE,
            half_precision=HALF_PRECISION,
            detect_start=DETECT_START,
            detect_end=DETECT_END,
        ),
        track=config.track,
        undistort=config.undistort,
        transform=config.transform,
        gui=config.gui,
    )
