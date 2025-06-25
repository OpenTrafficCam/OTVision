from dataclasses import dataclass
from unittest.mock import Mock

from OTVision.application.config import Config, TrackConfig, _TrackIouConfig
from OTVision.application.track.update_current_track_config import (
    UpdateCurrentTrackConfig,
)

DEFAULT_CONFIG = Config()


NEW_PATHS = ["path/to/file1.otdet", "path/to/file2.otdet"]
NEW_RUN_CHAINED = False
NEW_SIGMA_L = 0.51
NEW_SIGMA_H = 0.52
NEW_SIGMA_IOU = 0.53
NEW_T_MIN = 11
NEW_T_MISS_MAX = 32
NEW_OVERWRITE = True


class TestUpdateCurrentTrackConfig:
    def test_update(self) -> None:
        given = setup(create_given())
        target = create_target(given)

        target.update(given.track_config)

        given.get_current_config.get.assert_called_once()
        given.update_current_config.update.assert_called_once_with(expected_config())


@dataclass(frozen=True)
class Given:
    get_current_config: Mock
    update_current_config: Mock
    track_config: TrackConfig


def create_given() -> Given:
    return Given(Mock(), Mock(), track_config())


def track_config() -> TrackConfig:
    return TrackConfig(
        paths=NEW_PATHS,
        run_chained=NEW_RUN_CHAINED,
        iou=_TrackIouConfig(
            sigma_l=NEW_SIGMA_L,
            sigma_h=NEW_SIGMA_H,
            sigma_iou=NEW_SIGMA_IOU,
            t_min=NEW_T_MIN,
            t_miss_max=NEW_T_MISS_MAX,
        ),
        overwrite=NEW_OVERWRITE,
    )


def expected_config() -> Config:
    return Config(
        log=DEFAULT_CONFIG.log,
        search_subdirs=DEFAULT_CONFIG.search_subdirs,
        default_filetype=DEFAULT_CONFIG.default_filetype,
        last_paths=DEFAULT_CONFIG.last_paths,
        convert=DEFAULT_CONFIG.convert,
        detect=DEFAULT_CONFIG.detect,
        track=track_config(),
        undistort=DEFAULT_CONFIG.undistort,
        transform=DEFAULT_CONFIG.transform,
        gui=DEFAULT_CONFIG.gui,
    )


def setup(given: Given) -> Given:
    given.get_current_config.get.return_value = DEFAULT_CONFIG
    return given


def create_target(given: Given) -> UpdateCurrentTrackConfig:
    return UpdateCurrentTrackConfig(
        get_current_config=given.get_current_config,
        update_current_config=given.update_current_config,
    )
