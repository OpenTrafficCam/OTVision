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
    "input_fps": {"passed": "--input_fps 30", "expected": 30},
    "fps_from_filename": {"passed": "--fps_from_filename", "expected": True},
    "overwrite": {"passed": "--overwrite", "expected": True},
    "delete_input": {"passed": "--no-delete_input", "expected": False},
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
    "input_fps": {"passed": "--input_fps 25", "expected": 25},
    "fps_from_filename": {"passed": "--no-fps_from_filename", "expected": False},
    "overwrite": {"passed": "--no-overwrite", "expected": False},
    "delete_input": {"passed": "--delete_input", "expected": True},
    "config": {"passed": ""},
}

TEST_DATA_PARAMS_FROM_DEFAULT_CONFIG = {
    "paths": {
        "passed": "-p /usr/local/bin",
        "expected": [
            Path("/usr/local/bin"),
        ],
    },
    "input_fps": {"passed": "", "expected": cwd_config["CONVERT"]["INPUT_FPS"]},
    "fps_from_filename": {
        "passed": "",
        "expected": cwd_config["CONVERT"]["FPS_FROM_FILENAME"],
    },
    "overwrite": {"passed": "", "expected": cwd_config["CONVERT"]["OVERWRITE"]},
    "delete_input": {"passed": "", "expected": cwd_config["CONVERT"]["DELETE_INPUT"]},
    "config": {"passed": ""},
}

TEST_DATA_PARAMS_FROM_CUSTOM_CONFIG = {
    "paths": {
        "passed": "-p /usr/local/bin",
        "expected": [
            Path("/usr/local/bin"),
        ],
    },
    "input_fps": {"passed": "", "expected": custom_config["CONVERT"]["INPUT_FPS"]},
    "fps_from_filename": {
        "passed": "",
        "expected": custom_config["CONVERT"]["FPS_FROM_FILENAME"],
    },
    "overwrite": {"passed": "", "expected": custom_config["CONVERT"]["OVERWRITE"]},
    "delete_input": {
        "passed": "",
        "expected": custom_config["CONVERT"]["DELETE_INPUT"],
    },
    "config": {"passed": f"--config {CUSTOM_CONFIG_FILE}"},
}


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
                "convert.py",
                *test_data["paths"]["passed"].split(),
                *test_data["input_fps"]["passed"].split(),
                *test_data["fps_from_filename"]["passed"].split(),
                *test_data["overwrite"]["passed"].split(),
                *test_data["delete_input"]["passed"].split(),
                *test_data["config"]["passed"].split(),
            ]

            convert_cli(argv=list(filter(None, command)))

            mock_convert.assert_called_once_with(
                paths=test_data["paths"]["expected"],
                input_fps=test_data["input_fps"]["expected"],
                fps_from_filename=test_data["fps_from_filename"]["expected"],
                overwrite=test_data["overwrite"]["expected"],
                delete_input=test_data["delete_input"]["expected"],
            )
