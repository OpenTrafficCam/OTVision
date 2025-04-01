from pathlib import Path
from unittest.mock import Mock

from OTVision.application.config import Config, TrackConfig, _LogConfig, _TrackIouConfig
from OTVision.application.track.update_track_config_with_cli_args import (
    UpdateTrackConfigWithCliArgs,
)
from OTVision.domain.cli import TrackCliArgs

PATHS = ["file1.otdet", "file2.otdet"]
CONFIG_FILE = Path("config.yaml")
LOGFILE = Path("logfile.log")
LOGFILE_OVERWRITE = True
LOG_LEVEL_CONSOLE = "DEBUG"
LOG_LEVEL_FILE = "DEBUG"
OVERWRITE = True
SIGMA_L = 0.3
SIGMA_H = 0.5
SIGMA_IOU = 0.55
T_MIN = 10
T_MISS_MAX = 63


class TestUpdateTrackConfigWithCliArgs:
    def test_update(self) -> None:
        """
        #Requirement https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7148
        """  # noqa
        given = create_get_track_cli_args()
        given_config = custom_config()
        target = UpdateTrackConfigWithCliArgs(given)

        actual = target.update(given_config)
        assert actual == expected_config()
        given.get.assert_called_once()


def custom_config() -> Config:
    track_config = TrackConfig(
        iou=_TrackIouConfig(
            sigma_l=0.0, sigma_h=0.0, sigma_iou=0.0, t_min=0, t_miss_max=0
        )
    )
    return Config(track=track_config)


def cli_args() -> TrackCliArgs:
    return TrackCliArgs(
        paths=PATHS,
        config_file=CONFIG_FILE,
        logfile=LOGFILE,
        logfile_overwrite=LOGFILE_OVERWRITE,
        log_level_file=LOG_LEVEL_FILE,
        log_level_console=LOG_LEVEL_CONSOLE,
        overwrite=OVERWRITE,
        sigma_l=SIGMA_L,
        sigma_h=SIGMA_H,
        sigma_iou=SIGMA_IOU,
        t_min=T_MIN,
        t_miss_max=T_MISS_MAX,
    )


def create_get_track_cli_args() -> Mock:
    get_cli_args = Mock()
    get_cli_args.get.return_value = cli_args()
    return get_cli_args


def expected_config() -> Config:
    config = custom_config()
    return Config(
        log=_LogConfig(
            log_level_file=LOG_LEVEL_FILE, log_level_console=LOG_LEVEL_CONSOLE
        ),
        search_subdirs=config.search_subdirs,
        default_filetype=config.default_filetype,
        last_paths=config.last_paths,
        convert=config.convert,
        detect=config.detect,
        track=TrackConfig(
            paths=PATHS,
            run_chained=config.track.run_chained,
            iou=_TrackIouConfig(
                sigma_l=SIGMA_L,
                sigma_h=SIGMA_H,
                sigma_iou=SIGMA_IOU,
                t_min=T_MIN,
                t_miss_max=T_MISS_MAX,
            ),
            overwrite=OVERWRITE,
        ),
        undistort=config.undistort,
        transform=config.transform,
        gui=config.gui,
    )
