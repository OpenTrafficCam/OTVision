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
        "passed": f"-p ./ ./{CUSTOM_CONFIG_FILE}",
        "expected": [
            Path("./"),
            Path(f"./{CUSTOM_CONFIG_FILE}"),
        ],
    },
    "weights": {"passed": "--weights yolov5l", "expected": "yolov5l"},
    "conf": {"passed": "--conf 0.5", "expected": 0.5},
    "iou": {"passed": "--iou 0.55", "expected": 0.55},
    "chunksize": {"passed": "--chunksize 10", "expected": 10},
    "imagesize": {"passed": "--imagesize 1240", "expected": 1240},
    "half_precision": {"passed": "--half", "expected": True},
    "force_reload": {"passed": "--force", "expected": True},
    "overwrite": {"passed": "--overwrite", "expected": True},
    "config": {"passed": ""},
}

TEST_DATA_ALL_PARAMS_FROM_CLI_2 = {
    "paths": {
        "passed": f"-p ./ ./{CUSTOM_CONFIG_FILE}",
        "expected": [
            Path("./"),
            Path(f"./{CUSTOM_CONFIG_FILE}"),
        ],
    },
    "weights": {"passed": "--weights yolov5x", "expected": "yolov5x"},
    "conf": {"passed": "--conf 0.6", "expected": 0.6},
    "iou": {"passed": "--iou 0.65", "expected": 0.65},
    "chunksize": {"passed": "--chunksize 20", "expected": 20},
    "imagesize": {"passed": "--imagesize 320", "expected": 320},
    "half_precision": {"passed": "--no-half", "expected": False},
    "force_reload": {"passed": "--no-force", "expected": False},
    "overwrite": {"passed": "--no-overwrite", "expected": False},
    "config": {"passed": ""},
}

TEST_DATA_PARAMS_FROM_DEFAULT_CONFIG = {
    "paths": {"passed": "-p ./", "expected": [Path("./")]},
    "weights": {"passed": "", "expected": cwd_config["DETECT"]["YOLO"]["WEIGHTS"]},
    "conf": {"passed": "", "expected": cwd_config["DETECT"]["YOLO"]["CONF"]},
    "iou": {"passed": "", "expected": cwd_config["DETECT"]["YOLO"]["IOU"]},
    "chunksize": {"passed": "", "expected": cwd_config["DETECT"]["YOLO"]["CHUNKSIZE"]},
    "imagesize": {"passed": "", "expected": cwd_config["DETECT"]["YOLO"]["IMGSIZE"]},
    "half_precision": {
        "passed": "",
        "expected": cwd_config["DETECT"]["HALF_PRECISION"],
    },
    "force_reload": {
        "passed": "",
        "expected": cwd_config["DETECT"]["FORCE_RELOAD_TORCH_HUB_CACHE"],
    },
    "overwrite": {"passed": "", "expected": cwd_config["DETECT"]["OVERWRITE"]},
    "config": {"passed": ""},
}

TEST_DATA_PARAMS_FROM_CUSTOM_CONFIG = {
    "paths": {
        "passed": "",
        "expected": [
            Path(custom_config["DETECT"]["PATHS"][0]),
            Path(custom_config["DETECT"]["PATHS"][1]),
        ],
    },
    "weights": {"passed": "", "expected": custom_config["DETECT"]["YOLO"]["WEIGHTS"]},
    "conf": {"passed": "", "expected": custom_config["DETECT"]["YOLO"]["CONF"]},
    "iou": {"passed": "", "expected": custom_config["DETECT"]["YOLO"]["IOU"]},
    "chunksize": {
        "passed": "",
        "expected": custom_config["DETECT"]["YOLO"]["CHUNKSIZE"],
    },
    "imagesize": {"passed": "", "expected": custom_config["DETECT"]["YOLO"]["IMGSIZE"]},
    "half_precision": {
        "passed": "",
        "expected": custom_config["DETECT"]["HALF_PRECISION"],
    },
    "force_reload": {
        "passed": "",
        "expected": custom_config["DETECT"]["FORCE_RELOAD_TORCH_HUB_CACHE"],
    },
    "overwrite": {"passed": "", "expected": custom_config["DETECT"]["OVERWRITE"]},
    "config": {"passed": f"--config {CUSTOM_CONFIG_FILE}"},
}

TEST_FAIL_DATA = [
    {"passed": "--conf foo", "error_msg_part": "invalid float value: 'foo'"},
    {"passed": "--iou foo", "error_msg_part": "invalid float value: 'foo'"},
    {"passed": "--chunksize 2.2", "error_msg_part": "invalid int value: '2.2'"},
    {"passed": "--imagesize 2.2", "error_msg_part": "invalid int value: '2.2'"},
    {"passed": "--half foo", "error_msg_part": "unrecognized arguments"},
    {"passed": "--force foo", "error_msg_part": "unrecognized arguments"},
    {"passed": "--overwrite foo", "error_msg_part": "unrecognized arguments"},
    {
        "passed": "--no-weights",
        "error_msg_part": "unrecognized arguments: --no-weights",
    },
    {
        "passed": "--no-conf",
        "error_msg_part": "unrecognized arguments: --no-conf",
    },
    {
        "passed": "--no-iou",
        "error_msg_part": "unrecognized arguments: --no-iou",
    },
    {
        "passed": "--no-chunksize",
        "error_msg_part": "unrecognized arguments: --no-chunksize",
    },
    {
        "passed": "--no-imagesize",
        "error_msg_part": "unrecognized arguments: --no-imagesize",
    },
]


@pytest.fixture()
def detect_cli() -> Callable:
    """Imports and returns the main from the detect.py cli script in the root dir.

    Returns:
        Callable: main from the detect.py cli script in the root dir
    """
    from detect import main as detect_cli

    return detect_cli


@pytest.fixture()
def detect() -> Callable:
    """Imports and returns the main from OTVision.detect.detect.py

    Returns:
        Callable: main from OTVision.detect.detect.py
    """
    from OTVision import detect

    return detect


class TestDetectCLI:
    @pytest.mark.parametrize(
        argnames="test_data",
        argvalues=[
            TEST_DATA_ALL_PARAMS_FROM_CLI_1,
            TEST_DATA_ALL_PARAMS_FROM_CLI_2,
            TEST_DATA_PARAMS_FROM_DEFAULT_CONFIG,
            TEST_DATA_PARAMS_FROM_CUSTOM_CONFIG,
        ],
    )
    def test_pass_detect_cli(
        self, test_data: dict, detect_cli: Callable, detect: Callable
    ) -> None:
        detect = mock.create_autospec(detect)

        with patch("OTVision.detect") as mock_detect:
            command = [
                "detect.py",
                *test_data["paths"]["passed"].split(),
                *test_data["weights"]["passed"].split(),
                *test_data["conf"]["passed"].split(),
                *test_data["iou"]["passed"].split(),
                *test_data["chunksize"]["passed"].split(),
                *test_data["imagesize"]["passed"].split(),
                *test_data["half_precision"]["passed"].split(),
                *test_data["force_reload"]["passed"].split(),
                *test_data["overwrite"]["passed"].split(),
                *test_data["config"]["passed"].split(),
            ]

            detect_cli(argv=list(filter(None, command)))

            mock_detect.assert_called_once_with(
                paths=test_data["paths"]["expected"],
                weights=test_data["weights"]["expected"],
                conf=test_data["conf"]["expected"],
                iou=test_data["iou"]["expected"],
                size=test_data["imagesize"]["expected"],
                chunksize=test_data["chunksize"]["expected"],
                half_precision=test_data["half_precision"]["expected"],
                force_reload_torch_hub_cache=test_data["force_reload"]["expected"],
                overwrite=test_data["overwrite"]["expected"],
            )

    @pytest.mark.parametrize(argnames="test_fail_data", argvalues=TEST_FAIL_DATA)
    def test_fail_wrong_types_passed_to_detect_cli(
        self,
        detect_cli: Callable,
        detect: Callable,
        capsys: pytest.CaptureFixture,
        test_fail_data: dict,
    ) -> None:
        detect = mock.create_autospec(detect)

        with patch("OTVision.detect"):
            with pytest.raises(SystemExit) as e:
                command = ["detect.py", *test_fail_data["passed"].split()]
                detect_cli(argv=list(filter(None, command)))
            assert e.value.code == 2
            captured = capsys.readouterr()
            assert test_fail_data["error_msg_part"] in captured.err

    @pytest.mark.parametrize("passed", argvalues=["--config foo", "--paths foo"])
    def test_fail_not_existing_path_passed_to_detect_cli(
        self, detect: Callable, detect_cli: Callable, passed: str
    ) -> None:
        detect = mock.create_autospec(detect)

        with patch("OTVision.detect"):
            with pytest.raises(FileNotFoundError):
                command = ["detect.py", *passed.split()]
                detect_cli(argv=list(filter(None, command)))

    def test_fail_no_paths_passed_to_detect_cli(
        self, detect: Callable, detect_cli: Callable
    ) -> None:
        detect = mock.create_autospec(detect)

        with patch("OTVision.detect"):
            error_msg = (
                "No paths have been passed as command line args."
                + "No paths have been defined in the user config."
            )
            with pytest.raises(OSError, match=error_msg):
                detect_cli(argv=["detect.py"])
