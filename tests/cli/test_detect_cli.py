import unittest.mock as mock
from pathlib import Path
from typing import Callable
from unittest.mock import Mock, patch

import pytest
import yaml

from OTVision.config import (
    CONF,
    DEFAULT_EXPECTED_DURATION,
    DETECT,
    HALF_PRECISION,
    IMG_SIZE,
    IOU,
    OVERWRITE,
    PATHS,
    WEIGHTS,
    YOLO,
)
from OTVision.domain.cli import CliParseError

EXPECTED_DURATION = DEFAULT_EXPECTED_DURATION
INPUT_EXPECTED_DURATION = int(EXPECTED_DURATION.total_seconds())

CUSTOM_CONFIG_FILE = r"tests/cli/custom_cli_test_config.yaml"
with open(CUSTOM_CONFIG_FILE, "r") as file:
    custom_config = yaml.safe_load(file)

CWD_CONFIG_FILE = r"user_config.otvision.yaml"
with open(CWD_CONFIG_FILE, "r") as file:
    cwd_config = yaml.safe_load(file)

LOGFILE_OVERWRITE_CMD = "--logfile_overwrite"
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
    "expected_duration": {
        PASSED: f"--expected_duration {INPUT_EXPECTED_DURATION}",
        EXPECTED: EXPECTED_DURATION,
    },
    "overwrite": {PASSED: "--overwrite", EXPECTED: True},
    "config": {PASSED: ""},
    "detect_start": {PASSED: "--detect_start 300", EXPECTED: 300},
    "detect_end": {PASSED: "--detect_end 600", EXPECTED: 600},
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
    "expected_duration": {
        PASSED: f"--expected_duration {INPUT_EXPECTED_DURATION}",
        EXPECTED: EXPECTED_DURATION,
    },
    "overwrite": {PASSED: "--no-overwrite", EXPECTED: False},
    "detect_start": {PASSED: "--detect_start 300", EXPECTED: 300},
    "detect_end": {PASSED: "--detect_end 600", EXPECTED: 600},
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
    "expected_duration": {
        PASSED: f"--expected_duration {INPUT_EXPECTED_DURATION}",
        EXPECTED: EXPECTED_DURATION,
    },
    "overwrite": {PASSED: "", EXPECTED: cwd_config[DETECT][OVERWRITE]},
    "detect_start": {PASSED: "", EXPECTED: None},
    "detect_end": {PASSED: "", EXPECTED: None},
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
    "expected_duration": {
        PASSED: f"--expected_duration {INPUT_EXPECTED_DURATION}",
        EXPECTED: EXPECTED_DURATION,
    },
    "overwrite": {PASSED: "", EXPECTED: custom_config[DETECT][OVERWRITE]},
    "config": {PASSED: f"--config {CUSTOM_CONFIG_FILE}"},
    "detect_start": {PASSED: "", EXPECTED: None},
    "detect_end": {PASSED: "", EXPECTED: None},
}

required_arguments = (
    f"--expected_duration {INPUT_EXPECTED_DURATION} {LOGFILE_OVERWRITE_CMD}"
)
TEST_FAIL_DATA = [
    {
        PASSED: f"{required_arguments} --conf foo",
        "error_msg_part": "invalid float value: 'foo'",
    },
    {
        PASSED: f"{required_arguments} --iou foo",
        "error_msg_part": "invalid float value: 'foo'",
    },
    {
        PASSED: f"{required_arguments} --imagesize 2.2",
        "error_msg_part": "invalid int value: '2.2'",
    },
    {
        PASSED: f"{required_arguments} --half foo",
        "error_msg_part": "unrecognized arguments",
    },
    {
        PASSED: f"{required_arguments} --overwrite foo",
        "error_msg_part": "unrecognized arguments",
    },
    {
        PASSED: f"{required_arguments} --no-weights",
        "error_msg_part": "unrecognized arguments: --no-weights",
    },
    {
        PASSED: f"{required_arguments} --no-conf",
        "error_msg_part": "unrecognized arguments: --no-conf",
    },
    {
        PASSED: f"{required_arguments} --no-iou",
        "error_msg_part": "unrecognized arguments: --no-iou",
    },
    {
        PASSED: f"{required_arguments} --no-imagesize",
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
    ) -> None:
        mock_model = Mock()
        with patch("OTVision.detect") as mock_detect:
            with patch("detect.create_model") as mock_create_model:
                mock_create_model.return_value = mock_model
                command = [
                    *test_data["paths"][PASSED].split(),
                    *test_data["weights"][PASSED].split(),
                    *test_data["conf"][PASSED].split(),
                    *test_data["iou"][PASSED].split(),
                    *test_data["imagesize"][PASSED].split(),
                    *test_data["half_precision"][PASSED].split(),
                    *test_data["expected_duration"][PASSED].split(),
                    *test_data["overwrite"][PASSED].split(),
                    *test_data["config"][PASSED].split(),
                    *test_data["detect_start"][PASSED].split(),
                    *test_data["detect_end"][PASSED].split(),
                    LOGFILE_OVERWRITE_CMD,
                ]

                detect_cli(argv=list(filter(None, command)))

                mock_create_model.assert_called_once_with(
                    weights=test_data["weights"][EXPECTED],
                    confidence=test_data["conf"][EXPECTED],
                    iou=test_data["iou"][EXPECTED],
                    img_size=test_data["imagesize"][EXPECTED],
                    half_precision=test_data["half_precision"][EXPECTED],
                    normalized=False,
                )
                assert mock_create_model.call_count == 1

                assert mock_detect.call_args_list == [
                    mock.call(
                        model=mock_model,
                        paths=test_data["paths"][EXPECTED],
                        expected_duration=(test_data["expected_duration"][EXPECTED]),
                        overwrite=test_data["overwrite"][EXPECTED],
                        detect_start=test_data["detect_start"][EXPECTED],
                        detect_end=test_data["detect_end"][EXPECTED],
                    )
                ]

    @pytest.mark.parametrize(argnames="test_fail_data", argvalues=TEST_FAIL_DATA)
    def test_fail_wrong_types_passed_to_detect_cli(
        self,
        detect_cli: Callable,
        capsys: pytest.CaptureFixture,
        test_fail_data: dict,
    ) -> None:

        with patch("OTVision.detect"):
            with pytest.raises(SystemExit) as e:
                command = [*test_fail_data[PASSED].split()]
                detect_cli(argv=list(filter(None, command)))
            assert e.value.code == 2
            captured = capsys.readouterr()
            assert test_fail_data["error_msg_part"] in captured.err

    @pytest.mark.parametrize(PASSED, argvalues=["--config foo", "--paths foo"])
    def test_fail_not_existing_path_passed_to_detect_cli(
        self, detect_cli: Callable, passed: str
    ) -> None:
        with patch("OTVision.detect"):
            with pytest.raises(FileNotFoundError):
                command = required_arguments.split() + [*passed.split()]
                detect_cli(argv=list(filter(None, command)))

    def test_fail_no_paths_passed_to_detect_cli(self, detect_cli: Callable) -> None:
        with patch("OTVision.detect"):
            error_msg = (
                "No paths have been passed as command line args."
                + "No paths have been defined in the user config."
            )
            with pytest.raises(CliParseError, match=error_msg):
                detect_cli(argv=required_arguments.split())
