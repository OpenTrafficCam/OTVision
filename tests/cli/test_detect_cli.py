import unittest.mock as mock
from pathlib import Path
from typing import Callable
from unittest.mock import Mock, patch

import pytest
import yaml

from OTVision.config import (
    CONF,
    DETECT,
    HALF_PRECISION,
    IMG_SIZE,
    IOU,
    OVERWRITE,
    PATHS,
    WEIGHTS,
    YOLO,
)

CUSTOM_CONFIG_FILE = r"tests/cli/custom_cli_test_config.yaml"
with open(CUSTOM_CONFIG_FILE, "r") as file:
    custom_config = yaml.safe_load(file)

CWD_CONFIG_FILE = r"user_config.otvision.yaml"
with open(CWD_CONFIG_FILE, "r") as file:
    cwd_config = yaml.safe_load(file)

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
    "weights": {PASSED: "--weights yolov8l", EXPECTED: "yolov8l"},
    "conf": {PASSED: "--conf 0.5", EXPECTED: 0.5},
    "iou": {PASSED: "--iou 0.55", EXPECTED: 0.55},
    "imagesize": {PASSED: "--imagesize 1240", EXPECTED: 1240},
    "half_precision": {PASSED: "--half", EXPECTED: True},
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
    "weights": {PASSED: "--weights yolov8x", EXPECTED: "yolov8x"},
    "conf": {PASSED: "--conf 0.6", EXPECTED: 0.6},
    "iou": {PASSED: "--iou 0.65", EXPECTED: 0.65},
    "imagesize": {PASSED: "--imagesize 320", EXPECTED: 320},
    "half_precision": {PASSED: "--no-half", EXPECTED: False},
    "overwrite": {PASSED: "--no-overwrite", EXPECTED: False},
    "config": {PASSED: ""},
}

TEST_DATA_PARAMS_FROM_DEFAULT_CONFIG = {
    "paths": {PASSED: "-p ./", EXPECTED: [Path("./")]},
    "weights": {PASSED: "", EXPECTED: cwd_config[DETECT][YOLO][WEIGHTS]},
    "conf": {PASSED: "", EXPECTED: cwd_config[DETECT][YOLO][CONF]},
    "iou": {PASSED: "", EXPECTED: cwd_config[DETECT][YOLO][IOU]},
    "imagesize": {PASSED: "", EXPECTED: cwd_config[DETECT][YOLO][IMG_SIZE]},
    "half_precision": {
        PASSED: "",
        EXPECTED: cwd_config[DETECT][HALF_PRECISION],
    },
    "overwrite": {PASSED: "", EXPECTED: cwd_config[DETECT][OVERWRITE]},
    "config": {PASSED: ""},
}

TEST_DATA_PARAMS_FROM_CUSTOM_CONFIG = {
    "paths": {
        PASSED: "",
        EXPECTED: [
            Path(custom_config[DETECT][PATHS][0]),
            Path(custom_config[DETECT][PATHS][1]),
        ],
    },
    "weights": {PASSED: "", EXPECTED: custom_config[DETECT][YOLO][WEIGHTS]},
    "conf": {PASSED: "", EXPECTED: custom_config[DETECT][YOLO][CONF]},
    "iou": {PASSED: "", EXPECTED: custom_config[DETECT][YOLO][IOU]},
    "imagesize": {PASSED: "", EXPECTED: custom_config[DETECT][YOLO][IMG_SIZE]},
    "half_precision": {
        PASSED: "",
        EXPECTED: custom_config[DETECT][HALF_PRECISION],
    },
    "overwrite": {PASSED: "", EXPECTED: custom_config[DETECT][OVERWRITE]},
    "config": {PASSED: f"--config {CUSTOM_CONFIG_FILE}"},
}

TEST_FAIL_DATA = [
    {PASSED: "--conf foo", "error_msg_part": "invalid float value: 'foo'"},
    {PASSED: "--iou foo", "error_msg_part": "invalid float value: 'foo'"},
    {PASSED: "--imagesize 2.2", "error_msg_part": "invalid int value: '2.2'"},
    {PASSED: "--half foo", "error_msg_part": "unrecognized arguments"},
    {PASSED: "--overwrite foo", "error_msg_part": "unrecognized arguments"},
    {
        PASSED: "--no-weights",
        "error_msg_part": "unrecognized arguments: --no-weights",
    },
    {
        PASSED: "--no-conf",
        "error_msg_part": "unrecognized arguments: --no-conf",
    },
    {
        PASSED: "--no-iou",
        "error_msg_part": "unrecognized arguments: --no-iou",
    },
    {
        PASSED: "--no-imagesize",
        "error_msg_part": "unrecognized arguments: --no-imagesize",
    },
]


@pytest.fixture
def detect_cli() -> Callable:
    """Imports and returns the main from the detect.py cli script in the root dir.

    Returns:
        Callable: main from the detect.py cli script in the root dir
    """
    from detect import main as detect_cli

    return detect_cli


@pytest.fixture
def detect() -> Callable:
    """Imports and returns the main from OTVision.detect.detect.py

    Returns:
        Callable: main from OTVision.detect.detect.py
    """
    from OTVision import detect

    return detect


@pytest.fixture
def loadmodel() -> Callable:
    """Imports and returns the main from OTVision.detect.detect.py

    Returns:
        Callable: loadmodel from OTVision.detect.yolo.py
    """
    from OTVision.detect.yolo import loadmodel

    return loadmodel


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
        self,
        test_data: dict,
        detect_cli: Callable,
        detect: Callable,
        loadmodel: Callable,
    ) -> None:
        detect = mock.create_autospec(detect)
        loadmodel = mock.create_autospec(loadmodel)
        mock_model = Mock()

        with patch("OTVision.detect") as mock_detect:
            with patch("detect.loadmodel") as mock_loadmodel:
                mock_loadmodel.return_value = mock_model
                command = [
                    *test_data["paths"][PASSED].split(),
                    *test_data["weights"][PASSED].split(),
                    *test_data["conf"][PASSED].split(),
                    *test_data["iou"][PASSED].split(),
                    *test_data["imagesize"][PASSED].split(),
                    *test_data["half_precision"][PASSED].split(),
                    *test_data["overwrite"][PASSED].split(),
                    *test_data["config"][PASSED].split(),
                ]

                detect_cli(argv=list(filter(None, command)))

                mock_loadmodel.assert_any_call(
                    weights=test_data["weights"][EXPECTED],
                    confidence=test_data["conf"][EXPECTED],
                    iou=test_data["iou"][EXPECTED],
                    img_size=test_data["imagesize"][EXPECTED],
                    half_precision=test_data["half_precision"][EXPECTED],
                    normalized=False,
                )
                assert mock_loadmodel.call_count == 1

                mock_detect.assert_any_call(
                    model=mock_model,
                    paths=test_data["paths"][EXPECTED],
                    overwrite=test_data["overwrite"][EXPECTED],
                )
                assert mock_detect.call_count == 1

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
                command = [*test_fail_data[PASSED].split()]
                detect_cli(argv=list(filter(None, command)))
            assert e.value.code == 2
            captured = capsys.readouterr()
            assert test_fail_data["error_msg_part"] in captured.err

    @pytest.mark.parametrize(PASSED, argvalues=["--config foo", "--paths foo"])
    def test_fail_not_existing_path_passed_to_detect_cli(
        self, detect: Callable, detect_cli: Callable, passed: str
    ) -> None:
        detect = mock.create_autospec(detect)

        with patch("OTVision.detect"):
            with pytest.raises(FileNotFoundError):
                command = [*passed.split()]
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
                detect_cli(argv=[])
