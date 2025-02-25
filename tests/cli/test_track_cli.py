import unittest.mock as mock
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest
import yaml

from OTVision.config import (
    IOU,
    OVERWRITE,
    PATHS,
    SIGMA_H,
    SIGMA_IOU,
    SIGMA_L,
    T_MIN,
    T_MISS_MAX,
    TRACK,
)

CUSTOM_CONFIG_FILE = r"tests/cli/custom_cli_test_config.yaml"
with open(CUSTOM_CONFIG_FILE, "r") as file:
    custom_config = yaml.safe_load(file)

CWD_CONFIG_FILE = r"user_config.otvision.yaml"
with open(CWD_CONFIG_FILE, "r") as file:
    cwd_config = yaml.safe_load(file)

LOGFILE_OVERWRITE_CMD = "--logfile-overwrite"
PASSED: str = "passed"
EXPECTED: str = "expected"

TEST_DATA_ALL_PARAMS_FROM_CLI_1 = {
    "paths": {
        PASSED: f"-p ./ ./{CUSTOM_CONFIG_FILE}",
        EXPECTED: [
            Path("./"),
            Path(f"./{CUSTOM_CONFIG_FILE}"),
        ],
    },
    "sigma_h": {PASSED: "--sigma-h 0.37", EXPECTED: 0.37},
    "sigma_l": {PASSED: "--sigma-l 0.29", EXPECTED: 0.29},
    "sigma_iou": {PASSED: "--sigma-iou 0.36", EXPECTED: 0.36},
    "t_min": {PASSED: "--t-min 5", EXPECTED: 5},
    "t_miss_max": {PASSED: "--t-miss-max 38", EXPECTED: 38},
    "overwrite": {PASSED: "--overwrite", EXPECTED: True},
    "config": {PASSED: ""},
}

TEST_DATA_ALL_PARAMS_FROM_CLI_2 = {
    "paths": {
        PASSED: f"-p ./ ./{CUSTOM_CONFIG_FILE}",
        EXPECTED: [
            Path("./"),
            Path(f"./{CUSTOM_CONFIG_FILE}"),
        ],
    },
    "sigma_h": {PASSED: "--sigma-h 0.42", EXPECTED: 0.42},
    "sigma_l": {PASSED: "--sigma-l 0.34", EXPECTED: 0.34},
    "sigma_iou": {PASSED: "--sigma-iou 0.41", EXPECTED: 0.41},
    "t_min": {PASSED: "--t-min 7", EXPECTED: 7},
    "t_miss_max": {PASSED: "--t-miss-max 43", EXPECTED: 43},
    "overwrite": {PASSED: "--no-overwrite", EXPECTED: False},
    "config": {PASSED: ""},
}

TEST_DATA_PARAMS_FROM_DEFAULT_CONFIG = {
    "paths": {PASSED: "-p ./", EXPECTED: [Path("./")]},
    "sigma_h": {PASSED: "", EXPECTED: cwd_config[TRACK][IOU][SIGMA_H]},
    "sigma_l": {PASSED: "", EXPECTED: cwd_config[TRACK][IOU][SIGMA_L]},
    "sigma_iou": {PASSED: "", EXPECTED: cwd_config[TRACK][IOU][SIGMA_IOU]},
    "t_min": {PASSED: "", EXPECTED: cwd_config[TRACK][IOU][T_MIN]},
    "t_miss_max": {PASSED: "", EXPECTED: cwd_config[TRACK][IOU][T_MISS_MAX]},
    "overwrite": {PASSED: "", EXPECTED: cwd_config[TRACK][OVERWRITE]},
    "config": {PASSED: ""},
}

TEST_DATA_PARAMS_FROM_CUSTOM_CONFIG = {
    "paths": {
        PASSED: "",
        EXPECTED: [
            Path(custom_config[TRACK][PATHS][0]),
            Path(custom_config[TRACK][PATHS][1]),
        ],
    },
    "sigma_h": {PASSED: "", EXPECTED: custom_config[TRACK][IOU][SIGMA_H]},
    "sigma_l": {PASSED: "", EXPECTED: custom_config[TRACK][IOU][SIGMA_L]},
    "sigma_iou": {PASSED: "", EXPECTED: custom_config[TRACK][IOU][SIGMA_IOU]},
    "t_min": {PASSED: "", EXPECTED: custom_config[TRACK][IOU][T_MIN]},
    "t_miss_max": {
        PASSED: "",
        EXPECTED: custom_config[TRACK][IOU][T_MISS_MAX],
    },
    "overwrite": {PASSED: "", EXPECTED: custom_config[TRACK][OVERWRITE]},
    "config": {PASSED: f"--config {CUSTOM_CONFIG_FILE}"},
}

TEST_FAIL_DATA = [
    {PASSED: "--sigma-h foo", "error_msg_part": "invalid float value: 'foo'"},
    {PASSED: "--sigma-l foo", "error_msg_part": "invalid float value: 'foo'"},
    {PASSED: "--sigma-iou foo", "error_msg_part": "invalid float value: 'foo'"},
    {PASSED: "--t-min 2.2", "error_msg_part": "invalid int value: '2.2'"},
    {PASSED: "--t-miss-max 2.2", "error_msg_part": "invalid int value: '2.2'"},
    {PASSED: "--overwrite foo", "error_msg_part": "unrecognized arguments"},
    {
        PASSED: "--no-sigma-h",
        "error_msg_part": "unrecognized arguments: --no-sigma-h",
    },
    {
        PASSED: "--no-sigma-l",
        "error_msg_part": "unrecognized arguments: --no-sigma-l",
    },
    {
        PASSED: "--no-sigma-iou",
        "error_msg_part": "unrecognized arguments: --no-sigma-iou",
    },
    {
        PASSED: "--no-t-min",
        "error_msg_part": "unrecognized arguments: --no-t-min",
    },
    {
        PASSED: "--no-t-miss-max",
        "error_msg_part": "unrecognized arguments: --no-t-miss-max",
    },
]


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
                *test_data["paths"]["passed"].split(),
                *test_data["sigma_l"]["passed"].split(),
                *test_data["sigma_h"]["passed"].split(),
                *test_data["sigma_iou"]["passed"].split(),
                *test_data["t_min"]["passed"].split(),
                *test_data["t_miss_max"]["passed"].split(),
                *test_data["overwrite"]["passed"].split(),
                *test_data["config"]["passed"].split(),
                LOGFILE_OVERWRITE_CMD,
            ]

            track_cli(argv=list(filter(None, command)))

            mock_track.assert_called_once_with(
                paths=test_data["paths"][EXPECTED],
                sigma_l=test_data["sigma_l"][EXPECTED],
                sigma_h=test_data["sigma_h"][EXPECTED],
                sigma_iou=test_data["sigma_iou"][EXPECTED],
                t_min=test_data["t_min"][EXPECTED],
                t_miss_max=test_data["t_miss_max"][EXPECTED],
                overwrite=test_data["overwrite"][EXPECTED],
            )

    @pytest.mark.parametrize(argnames="test_fail_data", argvalues=TEST_FAIL_DATA)
    def test_fail_wrong_types_passed_to_track_cli(
        self,
        track_cli: Callable,
        track: Callable,
        capsys: pytest.CaptureFixture,
        test_fail_data: dict,
    ) -> None:
        track = mock.create_autospec(track)

        with patch("OTVision.track"):
            with pytest.raises(SystemExit) as e:
                command = [*test_fail_data["passed"].split()]
                track_cli(argv=list(filter(None, command)))
            assert e.value.code == 2
            captured = capsys.readouterr()
            assert test_fail_data["error_msg_part"] in captured.err

    @pytest.mark.parametrize("passed", argvalues=["--config foo", "--paths foo"])
    def test_fail_not_existing_path_passed_to_track_cli(
        self, track: Callable, track_cli: Callable, passed: str
    ) -> None:
        track = mock.create_autospec(track)

        with patch("OTVision.track"):
            with pytest.raises(FileNotFoundError):
                command = [*passed.split(), LOGFILE_OVERWRITE_CMD]
                track_cli(argv=list(filter(None, command)))

    def test_fail_no_paths_passed_to_track_cli(
        self, track: Callable, track_cli: Callable
    ) -> None:
        track = mock.create_autospec(track)

        with patch("OTVision.track"):
            error_msg = (
                "No paths have been passed as command line args."
                + "No paths have been defined in the user config."
            )
            with pytest.raises(OSError, match=error_msg):
                track_cli(argv=[LOGFILE_OVERWRITE_CMD])
