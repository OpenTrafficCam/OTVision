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
    Config,
    DetectConfig,
    YoloConfig,
)
from OTVision.domain.cli import CliParseError


def read_yaml(yaml_file: str) -> dict:
    with open(yaml_file, "r") as stream:
        return yaml.safe_load(stream)


EXPECTED_DURATION = DEFAULT_EXPECTED_DURATION
INPUT_EXPECTED_DURATION = int(EXPECTED_DURATION.total_seconds())

CUSTOM_CONFIG_FILE = r"tests/cli/custom_cli_test_config.yaml"
custom_config = read_yaml(CUSTOM_CONFIG_FILE)

CWD_CONFIG_FILE = r"user_config.otvision.yaml"
cwd_config = read_yaml(CWD_CONFIG_FILE)

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
    "weights": {PASSED: "--weights yolov8l", EXPECTED: "yolov8l"},
    "conf": {PASSED: "--conf 0.5", EXPECTED: 0.5},
    "iou": {PASSED: "--iou 0.55", EXPECTED: 0.55},
    "imagesize": {PASSED: "--imagesize 1240", EXPECTED: 1240},
    "half_precision": {PASSED: "--half", EXPECTED: True},
    "expected_duration": {
        PASSED: f"--expected-duration {INPUT_EXPECTED_DURATION}",
        EXPECTED: EXPECTED_DURATION,
    },
    "overwrite": {PASSED: "--overwrite", EXPECTED: True},
    "config": {PASSED: ""},
    "detect_start": {PASSED: "--detect-start 300", EXPECTED: 300},
    "detect_end": {PASSED: "--detect-end 600", EXPECTED: 600},
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
        PASSED: f"--expected-duration {INPUT_EXPECTED_DURATION}",
        EXPECTED: EXPECTED_DURATION,
    },
    "overwrite": {PASSED: "--no-overwrite", EXPECTED: False},
    "detect_start": {PASSED: "--detect-start 300", EXPECTED: 300},
    "detect_end": {PASSED: "--detect-end 600", EXPECTED: 600},
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
        PASSED: f"--expected-duration {INPUT_EXPECTED_DURATION}",
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
        PASSED: f"--expected-duration {INPUT_EXPECTED_DURATION}",
        EXPECTED: EXPECTED_DURATION,
    },
    "overwrite": {PASSED: "", EXPECTED: custom_config[DETECT][OVERWRITE]},
    "config": {PASSED: f"--config {CUSTOM_CONFIG_FILE}"},
    "detect_start": {PASSED: "", EXPECTED: None},
    "detect_end": {PASSED: "", EXPECTED: None},
}

required_arguments = (
    f"--expected-duration {INPUT_EXPECTED_DURATION} {LOGFILE_OVERWRITE_CMD}"
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
    @patch("detect.DetectBuilder.update_current_config")
    @patch("detect.DetectBuilder.build")
    def test_pass_detect_cli(
        self,
        mock_build: Mock,
        mock_update_current_config: Mock,
        test_data: dict,
        detect_cli: Callable,
    ) -> None:
        mock_otvision_detect = Mock()
        mock_build.return_value = mock_otvision_detect

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
        expected_config = create_expected_config_from_test_data(test_data)

        mock_update_current_config.update.assert_called_once_with(expected_config)
        mock_otvision_detect.start.assert_called_once()

    @pytest.mark.parametrize(argnames="test_fail_data", argvalues=TEST_FAIL_DATA)
    def test_fail_wrong_types_passed_to_detect_cli(
        self,
        detect_cli: Callable,
        capsys: pytest.CaptureFixture,
        test_fail_data: dict,
    ) -> None:

        with patch("detect.DetectBuilder.build"):
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
        with patch("detect.DetectBuilder.build"):
            with pytest.raises(FileNotFoundError):
                command = required_arguments.split() + [*passed.split()]
                detect_cli(argv=list(filter(None, command)))

    def test_fail_no_paths_passed_to_detect_cli(self, detect_cli: Callable) -> None:
        with patch("detect.DetectBuilder.build"):
            error_msg = (
                "No paths have been passed as command line args."
                + "No paths have been defined in the user config."
            )
            with pytest.raises(CliParseError, match=error_msg):
                detect_cli(argv=required_arguments.split())


def create_expected_config_from_test_data(test_data: dict) -> Config:
    """
    Create a Config object from the EXPECTED values of the provided test data
    dictionary.

    Args:
        test_data (dict): The dictionary containing EXPECTED values for configuration.

    Returns:
        Config: A Config object populated with the relevant configuration values.
    """

    if config_file_arg := test_data["config"].get(PASSED):
        default_config = Config.from_dict(read_yaml(config_file_arg.split()[1]))
    else:
        default_config = Config()

    # Map EXPECTED values to DetectConfig's relevant fields
    paths = test_data["paths"].get(EXPECTED, default_config.detect.paths)
    weights = test_data["weights"].get(
        EXPECTED, default_config.detect.yolo_config.weights
    )
    conf = test_data["conf"].get(EXPECTED, default_config.detect.yolo_config.conf)
    iou = test_data["iou"].get(EXPECTED, default_config.detect.yolo_config.iou)
    img_size = test_data["imagesize"].get(
        EXPECTED, default_config.detect.yolo_config.img_size
    )
    half_precision = test_data["half_precision"].get(
        EXPECTED, default_config.detect.half_precision
    )
    expected_duration = test_data["expected_duration"].get(
        EXPECTED, default_config.detect.expected_duration
    )
    overwrite = test_data["overwrite"].get(EXPECTED, default_config.detect.overwrite)
    detect_start = test_data["detect_start"].get(
        EXPECTED, default_config.detect.detect_start
    )
    detect_end = test_data["detect_end"].get(EXPECTED, default_config.detect.detect_end)

    # Ensure paths are converted to Path objects if necessary
    paths = [Path(p).expanduser() for p in paths]

    # Create YoloConfig using the extracted values
    yolo_config = YoloConfig(
        weights=weights,
        conf=conf,
        iou=iou,
        img_size=img_size,
        normalized=default_config.detect.yolo_config.normalized,
    )

    detect_config = DetectConfig(
        paths=paths,
        run_chained=default_config.detect.run_chained,
        yolo_config=yolo_config,
        expected_duration=expected_duration,
        overwrite=overwrite,
        half_precision=half_precision,
        detect_start=detect_start,
        detect_end=detect_end,
    )

    return Config(
        log=default_config.log,
        search_subdirs=default_config.search_subdirs,
        default_filetype=default_config.default_filetype,
        filetypes=default_config.filetypes,
        last_paths=default_config.last_paths,
        convert=default_config.convert,
        detect=detect_config,
        track=default_config.track,
        undistort=default_config.undistort,
        transform=default_config.transform,
        gui=default_config.gui,
    )
