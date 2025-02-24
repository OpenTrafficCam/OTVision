import unittest.mock as mock
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest
import yaml

from OTVision.config import (
    CONVERT,
    DELETE_INPUT,
    FPS_FROM_FILENAME,
    INPUT_FPS,
    OVERWRITE,
    PATHS,
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
    "input_fps": {PASSED: "--input-fps 30", EXPECTED: 30},
    "fps_from_filename": {PASSED: "--fps-from-filename", EXPECTED: True},
    "overwrite": {PASSED: "--overwrite", EXPECTED: True},
    "delete_input": {PASSED: "--no-delete-input", EXPECTED: False},
    "config": {PASSED: ""},
    "rotation": {PASSED: "--rotation 2", EXPECTED: 2},
}

TEST_DATA_ALL_PARAMS_FROM_CLI_2 = {
    "paths": {
        PASSED: f"-p ./ ./{CUSTOM_CONFIG_FILE}",
        EXPECTED: [
            Path("./"),
            Path(f"./{CUSTOM_CONFIG_FILE}"),
        ],
    },
    "input_fps": {PASSED: "--input-fps 25", EXPECTED: 25},
    "fps_from_filename": {PASSED: "--no-fps-from-filename", EXPECTED: False},
    "overwrite": {PASSED: "--no-overwrite", EXPECTED: False},
    "delete_input": {PASSED: "--delete-input", EXPECTED: True},
    "config": {PASSED: ""},
    "rotation": {PASSED: "--rotation 3", EXPECTED: 3},
}

TEST_DATA_PARAMS_FROM_DEFAULT_CONFIG = {
    "paths": {PASSED: "-p ./", EXPECTED: [Path("./")]},
    "input_fps": {PASSED: "", EXPECTED: cwd_config[CONVERT][INPUT_FPS]},
    "fps_from_filename": {
        PASSED: "",
        EXPECTED: cwd_config[CONVERT][FPS_FROM_FILENAME],
    },
    "overwrite": {PASSED: "", EXPECTED: cwd_config[CONVERT][OVERWRITE]},
    "delete_input": {PASSED: "", EXPECTED: cwd_config[CONVERT][DELETE_INPUT]},
    "config": {PASSED: ""},
    "rotation": {PASSED: "", EXPECTED: 0},
}

TEST_DATA_PARAMS_FROM_CUSTOM_CONFIG = {
    "paths": {
        PASSED: "",
        EXPECTED: [
            Path(custom_config[CONVERT][PATHS][0]),
            Path(custom_config[CONVERT][PATHS][1]),
        ],
    },
    "input_fps": {PASSED: "", EXPECTED: custom_config[CONVERT][INPUT_FPS]},
    "fps_from_filename": {
        PASSED: "",
        EXPECTED: custom_config[CONVERT][FPS_FROM_FILENAME],
    },
    "overwrite": {PASSED: "", EXPECTED: custom_config[CONVERT][OVERWRITE]},
    "delete_input": {
        PASSED: "",
        EXPECTED: custom_config[CONVERT][DELETE_INPUT],
    },
    "config": {PASSED: f"--config {CUSTOM_CONFIG_FILE}"},
    "rotation": {PASSED: "", EXPECTED: 0},
}

TEST_FAIL_DATA = [
    {PASSED: "--input-fps foo", "error_msg_part": "invalid float value: 'foo'"},
    {PASSED: "--fps-from-filename 20", "error_msg_part": "unrecognized arguments"},
    {PASSED: "--overwrite foo", "error_msg_part": "unrecognized arguments"},
    {PASSED: "--delete-input foo", "error_msg_part": "unrecognized arguments"},
    {
        PASSED: "--no-input-fps",
        "error_msg_part": "unrecognized arguments: --no-input-fps",
    },
]


@pytest.fixture()
def convert_cli() -> Callable:
    """Imports and returns the main from the convert.py cli script in the root dir.

    Returns:
        Callable: main from the convert.py cli script in the root dir
    """
    from convert import main as convert_cli

    return convert_cli


@pytest.fixture()
def convert() -> Callable:
    """Imports and returns the main from OTVision.convert.convert.py

    Returns:
        Callable: main from OTVision.convert.convert.py
    """
    from OTVision import convert

    return convert


class TestConvertCLI:
    @pytest.mark.parametrize(
        argnames="test_data",
        argvalues=[
            TEST_DATA_ALL_PARAMS_FROM_CLI_1,
            TEST_DATA_ALL_PARAMS_FROM_CLI_2,
            TEST_DATA_PARAMS_FROM_DEFAULT_CONFIG,
            TEST_DATA_PARAMS_FROM_CUSTOM_CONFIG,
        ],
    )
    def test_pass_convert_cli(
        self, test_data: dict, convert_cli: Callable, convert: Callable
    ) -> None:
        convert = mock.create_autospec(convert)

        with patch("OTVision.convert") as mock_convert:
            command = [
                *test_data["paths"][PASSED].split(),
                *test_data["input_fps"][PASSED].split(),
                *test_data["fps_from_filename"][PASSED].split(),
                *test_data["overwrite"][PASSED].split(),
                *test_data["delete_input"][PASSED].split(),
                *test_data["config"][PASSED].split(),
                *test_data["rotation"][PASSED].split(),
                LOGFILE_OVERWRITE_CMD,
            ]

            convert_cli(argv=list(filter(None, command)))

            mock_convert.assert_called_once_with(
                paths=test_data["paths"][EXPECTED],
                input_fps=test_data["input_fps"][EXPECTED],
                fps_from_filename=test_data["fps_from_filename"][EXPECTED],
                overwrite=test_data["overwrite"][EXPECTED],
                delete_input=test_data["delete_input"][EXPECTED],
                rotation=test_data["rotation"][EXPECTED],
            )

    @pytest.mark.parametrize(argnames="test_fail_data", argvalues=TEST_FAIL_DATA)
    def test_fail_wrong_types_passed_to_convert_cli(
        self,
        convert_cli: Callable,
        convert: Callable,
        capsys: pytest.CaptureFixture,
        test_fail_data: dict,
    ) -> None:
        convert = mock.create_autospec(convert)

        with patch("OTVision.convert"):
            with pytest.raises(SystemExit) as e:
                command = [*test_fail_data[PASSED].split()]
                convert_cli(argv=list(filter(None, command)))
            assert e.value.code == 2
            captured = capsys.readouterr()
            assert test_fail_data["error_msg_part"] in captured.err

    @pytest.mark.parametrize(PASSED, argvalues=["--config foo", "--paths foo"])
    def test_fail_not_existing_path_passed_to_convert_cli(
        self, convert: Callable, convert_cli: Callable, passed: str
    ) -> None:
        convert = mock.create_autospec(convert)

        with patch("OTVision.convert"):
            with pytest.raises(FileNotFoundError):
                command = [*passed.split(), LOGFILE_OVERWRITE_CMD]
                convert_cli(argv=list(filter(None, command)))

    def test_fail_no_paths_passed_to_convert_cli(
        self, convert: Callable, convert_cli: Callable
    ) -> None:
        convert = mock.create_autospec(convert)

        with patch("OTVision.convert"):
            error_msg = (
                "No paths have been passed as command line args."
                + "No paths have been defined in the user config."
            )
            with pytest.raises(OSError, match=error_msg):
                convert_cli(argv=[LOGFILE_OVERWRITE_CMD])
