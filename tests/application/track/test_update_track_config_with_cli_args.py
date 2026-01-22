from pathlib import Path
from unittest.mock import Mock

from OTVision.application.config import (
    Config,
    TrackConfig,
    _LogConfig,
    _TrackBoxmotConfig,
    _TrackIouConfig,
)
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


class TestUpdateBoxmotConfig:
    """Tests for BOXMOT config updates via CLI arguments."""

    def test_tracker_bytetrack_auto_enables_boxmot(self) -> None:
        """Test that --tracker bytetrack enables BOXMOT automatically."""
        cli_args = TrackCliArgs(
            paths=None,
            config_file=None,
            logfile=LOGFILE,
            logfile_overwrite=False,
            log_level_console=None,
            log_level_file=None,
            tracker="bytetrack",
        )
        get_cli_args = Mock()
        get_cli_args.get.return_value = cli_args
        config = Config(track=TrackConfig(boxmot=_TrackBoxmotConfig(enabled=False)))

        target = UpdateTrackConfigWithCliArgs(get_cli_args)
        result = target.update(config)

        assert result.track.boxmot.enabled is True
        assert result.track.boxmot.tracker_type == "bytetrack"

    def test_tracker_iou_disables_boxmot(self) -> None:
        """Test that --tracker iou disables BOXMOT."""
        cli_args = TrackCliArgs(
            paths=None,
            config_file=None,
            logfile=LOGFILE,
            logfile_overwrite=False,
            log_level_console=None,
            log_level_file=None,
            tracker="iou",
        )
        get_cli_args = Mock()
        get_cli_args.get.return_value = cli_args
        config = Config(track=TrackConfig(boxmot=_TrackBoxmotConfig(enabled=True)))

        target = UpdateTrackConfigWithCliArgs(get_cli_args)
        result = target.update(config)

        assert result.track.boxmot.enabled is False

    def test_cli_overrides_yaml_boxmot_config(self) -> None:
        """Test that CLI arguments override YAML config for BOXMOT."""
        cli_args = TrackCliArgs(
            paths=None,
            config_file=None,
            logfile=LOGFILE,
            logfile_overwrite=False,
            log_level_console=None,
            log_level_file=None,
            tracker="botsort",
            tracker_device="cuda:0",
            tracker_half_precision=True,
            tracker_reid_weights="/path/to/reid.pt",
            tracker_params={"track_buffer": 60},
        )
        get_cli_args = Mock()
        get_cli_args.get.return_value = cli_args
        yaml_boxmot = _TrackBoxmotConfig(
            enabled=False,
            tracker_type="bytetrack",
            device="cpu",
            half_precision=False,
            reid_weights=None,
            tracker_params={"other_param": 10},
        )
        config = Config(track=TrackConfig(boxmot=yaml_boxmot))

        target = UpdateTrackConfigWithCliArgs(get_cli_args)
        result = target.update(config)

        assert result.track.boxmot.enabled is True
        assert result.track.boxmot.tracker_type == "botsort"
        assert result.track.boxmot.device == "cuda:0"
        assert result.track.boxmot.half_precision is True
        assert result.track.boxmot.reid_weights == "/path/to/reid.pt"
        assert result.track.boxmot.tracker_params == {"track_buffer": 60}

    def test_no_cli_tracker_preserves_yaml_config(self) -> None:
        """Test that without --tracker, YAML config is preserved."""
        cli_args = TrackCliArgs(
            paths=None,
            config_file=None,
            logfile=LOGFILE,
            logfile_overwrite=False,
            log_level_console=None,
            log_level_file=None,
        )
        get_cli_args = Mock()
        get_cli_args.get.return_value = cli_args
        yaml_boxmot = _TrackBoxmotConfig(
            enabled=True,
            tracker_type="botsort",
            device="cuda:0",
            half_precision=True,
            reid_weights="/yaml/reid.pt",
            tracker_params={"yaml_param": 5},
        )
        config = Config(track=TrackConfig(boxmot=yaml_boxmot))

        target = UpdateTrackConfigWithCliArgs(get_cli_args)
        result = target.update(config)

        assert result.track.boxmot.enabled is True
        assert result.track.boxmot.tracker_type == "botsort"
        assert result.track.boxmot.device == "cuda:0"
        assert result.track.boxmot.half_precision is True
        assert result.track.boxmot.reid_weights == "/yaml/reid.pt"
        assert result.track.boxmot.tracker_params == {"yaml_param": 5}
