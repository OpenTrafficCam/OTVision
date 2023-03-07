import unittest.mock as mock
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest
import yaml

CUSTOM_CONFIG_FILE = r"tests/cli/custom_cli_test_config.yaml"
with open(CUSTOM_CONFIG_FILE, "r") as file:
    custom_config = yaml.safe_load(file)

CWD_CONFIG_FILE = r"user_config.otvision.yaml"
with open(CWD_CONFIG_FILE, "r") as file:
    cwd_config = yaml.safe_load(file)

TEST_DATA_ALL_PARAMS_FROM_CLI_1 = {
    "paths": {
        "passed": "-p /usr/local/bin C:/Python/Scripts",
        "expected": [
            Path("/usr/local/bin"),
            Path("C:/Python/Scripts"),
        ],
    },
    "sigma_h": {"passed": "--sigma_h 0.37", "expected": 0.37},
    "sigma_l": {"passed": "--sigma_l 0.29", "expected": 0.29},
    "sigma_iou": {"passed": "--sigma_iou 0.36", "expected": 0.36},
    "t_min": {"passed": "--t_min 5", "expected": 5},
    "t_miss_max": {"passed": "--t_miss_max 38", "expected": 38},
    "overwrite": {"passed": "--overwrite", "expected": True},
    "config": {"passed": ""},
}

TEST_DATA_ALL_PARAMS_FROM_CLI_2 = {
    "paths": {
        "passed": "-p /usr/local/file.ext C:/Python/file.ext",
        "expected": [
            Path("/usr/local/file.ext"),
            Path("C:/Python/file.ext"),
        ],
    },
    "sigma_h": {"passed": "--sigma_h 0.42", "expected": 0.42},
    "sigma_l": {"passed": "--sigma_l 0.34", "expected": 0.34},
    "sigma_iou": {"passed": "--sigma_iou 0.41", "expected": 0.41},
    "t_min": {"passed": "--t_min 7", "expected": 7},
    "t_miss_max": {"passed": "--t_miss_max 43", "expected": 43},
    "overwrite": {"passed": "--no-overwrite", "expected": False},
    "config": {"passed": ""},
}

TEST_DATA_PARAMS_FROM_DEFAULT_CONFIG = {
    "paths": {
        "passed": "-p /usr/local/bin",
        "expected": [
            Path("/usr/local/bin"),
        ],
    },
    "sigma_h": {"passed": "", "expected": cwd_config["TRACK"]["IOU"]["SIGMA_H"]},
    "sigma_l": {"passed": "", "expected": cwd_config["TRACK"]["IOU"]["SIGMA_L"]},
    "sigma_iou": {"passed": "", "expected": cwd_config["TRACK"]["IOU"]["SIGMA_IOU"]},
    "t_min": {"passed": "", "expected": cwd_config["TRACK"]["IOU"]["T_MIN"]},
    "t_miss_max": {"passed": "", "expected": cwd_config["TRACK"]["IOU"]["T_MISS_MAX"]},
    "overwrite": {"passed": "", "expected": cwd_config["TRACK"]["OVERWRITE"]},
    "config": {"passed": ""},
}

TEST_DATA_PARAMS_FROM_CUSTOM_CONFIG = {
    "paths": {
        "passed": "",
        "expected": [
            Path(custom_config["TRACK"]["PATHS"][0]),
            Path(custom_config["TRACK"]["PATHS"][1]),
        ],
    },
    "sigma_h": {"passed": "", "expected": custom_config["TRACK"]["IOU"]["SIGMA_H"]},
    "sigma_l": {"passed": "", "expected": custom_config["TRACK"]["IOU"]["SIGMA_L"]},
    "sigma_iou": {"passed": "", "expected": custom_config["TRACK"]["IOU"]["SIGMA_IOU"]},
    "t_min": {"passed": "", "expected": custom_config["TRACK"]["IOU"]["T_MIN"]},
    "t_miss_max": {
        "passed": "",
        "expected": custom_config["TRACK"]["IOU"]["T_MISS_MAX"],
    },
    "overwrite": {"passed": "", "expected": custom_config["TRACK"]["OVERWRITE"]},
    "config": {"passed": f"--config {CUSTOM_CONFIG_FILE}"},
}


@pytest.fixture()
def track_cli() -> Callable:
    """Imports and returns the main from the track.py cli script in the root dir.

    Returns:
        Callable: main from the track.py cli script in the root dir
    """
    from track import main as track_cli

    return track_cli


@pytest.fixture()
def track() -> Callable:
    """Imports and returns the main from OTVision.track.track.py

    Returns:
        Callable: main from OTVision.track.track.py
    """
    from OTVision import track

    return track


class TestTrackCLI:
    @pytest.mark.parametrize(
        argnames="test_data",
        argvalues=[
            TEST_DATA_ALL_PARAMS_FROM_CLI_1,
            TEST_DATA_ALL_PARAMS_FROM_CLI_2,
            TEST_DATA_PARAMS_FROM_DEFAULT_CONFIG,
            TEST_DATA_PARAMS_FROM_CUSTOM_CONFIG,
        ],
    )
    def test_pass_track_cli(
        self, test_data: dict, track_cli: Callable, track: Callable
    ) -> None:
        track = mock.create_autospec(track)

        with patch("OTVision.track") as mock_track:
            command = [
                "track.py",
                *test_data["paths"]["passed"].split(),
                *test_data["sigma_l"]["passed"].split(),
                *test_data["sigma_h"]["passed"].split(),
                *test_data["sigma_iou"]["passed"].split(),
                *test_data["t_min"]["passed"].split(),
                *test_data["t_miss_max"]["passed"].split(),
                *test_data["overwrite"]["passed"].split(),
                *test_data["config"]["passed"].split(),
            ]

            track_cli(argv=list(filter(None, command)))

            mock_track.assert_called_once_with(
                paths=test_data["paths"]["expected"],
                sigma_l=test_data["sigma_l"]["expected"],
                sigma_h=test_data["sigma_h"]["expected"],
                sigma_iou=test_data["sigma_iou"]["expected"],
                t_min=test_data["t_min"]["expected"],
                t_miss_max=test_data["t_miss_max"]["expected"],
                overwrite=test_data["overwrite"]["expected"],
            )
