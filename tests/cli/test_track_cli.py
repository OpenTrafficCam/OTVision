from dataclasses import replace
from pathlib import Path
from typing import Callable
from unittest.mock import AsyncMock, Mock, patch

import pytest

from OTVision.application.config import (
    IOU,
    OVERWRITE,
    PATHS,
    SIGMA_H,
    SIGMA_IOU,
    SIGMA_L,
    T_MIN,
    T_MISS_MAX,
    TRACK,
    Config,
    TrackConfig,
    _TrackIouConfig,
)
from OTVision.application.config_parser import ConfigParser, InvalidOtvisionConfigError
from OTVision.domain.cli import CliParseError
from OTVision.plugin.yaml_serialization import YamlDeserializer
from OTVision.track.builder import TrackBuilder

YAML_DESERIALIZER = YamlDeserializer()
CONFIG_PARSER = ConfigParser(YAML_DESERIALIZER)

CUSTOM_CONFIG_FILE = r"tests/cli/custom_cli_test_config.yaml"
custom_config = YAML_DESERIALIZER.deserialize(Path(CUSTOM_CONFIG_FILE))

CWD_CONFIG_FILE = r"user_config.otvision.yaml"
cwd_config = YAML_DESERIALIZER.deserialize(Path(CWD_CONFIG_FILE))

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


class TestTrackCLI:
    @pytest.mark.parametrize(
        "configured, selected", [("iou", "botsort"), ("botsort", "iou")]
    )
    def test_tracker_cli_selection_overrides_existing_config(
        self, tmp_path: Path, configured: str, selected: str
    ) -> None:
        config_file = tmp_path / "user_config.otvision.yaml"
        config_file.write_text(
            f"TRACK:\n  TRACKER_TYPE: {configured}\n"
            "  IOU:\n    T_MIN: 3\n    T_MISS_MAX: 25\n"
            "  BOT_SORT:\n    T_MIN: 12\n    T_MISS_MAX: 40\n"
        )
        builder = TrackBuilder(
            argv=["--config", str(config_file), "--tracker", selected]
        )
        cli_args = builder.get_track_cli_args.get()
        config = builder.get_config.get(cli_args)

        updated = builder.update_track_config_with_cli_args.update(config)

        assert updated.track.tracker_type == selected
        assert updated.track.iou == config.track.iou
        assert updated.track.bot_sort == config.track.bot_sort
        expected_lifecycle = (12, 40) if selected == "botsort" else (3, 25)
        assert (updated.track.t_min, updated.track.t_miss_max) == expected_lifecycle

    def test_configured_tracker_is_kept_and_checked_without_cli_selection(
        self, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "user_config.otvision.yaml"
        config_file.write_text("TRACK:\n  TRACKER_TYPE: iou\n")
        builder = TrackBuilder(argv=["--config", str(config_file)])
        cli_args = builder.get_track_cli_args.get()
        config = builder.get_config.get(cli_args)
        # The parser rejects an unknown type before it can reach the override step.
        config = replace(config, track=replace(config.track, tracker_type="unknown"))

        with pytest.raises(
            InvalidOtvisionConfigError, match=r"'unknown'.*options: iou, botsort"
        ):
            builder.update_track_config_with_cli_args.update(config)

    def test_reject_invalid_tracker_cli_selection(
        self, track_cli: Callable, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "user_config.otvision.yaml"
        config_file.write_text("TRACK:\n  TRACKER_TYPE: iou\n")

        with pytest.raises(
            InvalidOtvisionConfigError, match=r"'unknown'.*options: iou, botsort"
        ):
            track_cli(argv=["--config", str(config_file), "--tracker", "unknown"])

    @pytest.mark.parametrize("from_cli", [True, False], ids=["cli", "yaml"])
    @pytest.mark.parametrize(
        "option", ["--sigma-l", "--sigma-h", "--sigma-iou", "--t-min", "--t-miss-max"]
    )
    def test_reject_iou_options_with_bot_sort(
        self, track_cli: Callable, tmp_path: Path, from_cli: bool, option: str
    ) -> None:
        config_file = tmp_path / "user_config.otvision.yaml"
        configured = "iou" if from_cli else "botsort"
        config_file.write_text(f"TRACK:\n  TRACKER_TYPE: {configured}\n")
        selector_args = ["--tracker", "botsort"] if from_cli else []

        with pytest.raises(InvalidOtvisionConfigError, match=rf"{option}.*BoT-SORT"):
            track_cli(
                argv=[
                    "--config",
                    str(config_file),
                    *selector_args,
                    option,
                    "0",
                    LOGFILE_OVERWRITE_CMD,
                ]
            )

    def test_cli_tracker_selection_revalidates_stream_flush_buffer(
        self, track_cli: Callable, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "user_config.otvision.yaml"
        config_file.write_text(
            "TRACK:\n  TRACKER_TYPE: iou\n"
            "STREAM:\n"
            "  NAME: camera\n"
            "  SOURCE: rtsp://camera\n"
            "  SAVE_DIR: .\n"
            "  FLUSH_BUFFER_SIZE: 60\n"
        )

        with pytest.raises(InvalidOtvisionConfigError, match=r"t_miss_max \(60\)"):
            track_cli(
                argv=[
                    "--config",
                    str(config_file),
                    "--tracker",
                    "botsort",
                    LOGFILE_OVERWRITE_CMD,
                ]
            )

    @patch("track.TrackBuilder.update_current_config")
    @patch("track.TrackBuilder.build")
    def test_iou_cli_override_preserves_yaml_bot_sort_config(
        self,
        mock_build: Mock,
        mock_update_current_config: Mock,
        track_cli: Callable,
        tmp_path: Path,
    ) -> None:
        mock_build.return_value.start = AsyncMock()
        config_file = tmp_path / "user_config.otvision.yaml"
        config_file.write_text("TRACK:\n  BOT_SORT:\n    T_MIN: 12\n")

        track_cli(
            argv=["--config", str(config_file), "--t-min", "3", LOGFILE_OVERWRITE_CMD]
        )

        config = mock_update_current_config.update.call_args.kwargs["config"]
        assert config.track.t_min == 3
        assert config.track.bot_sort.t_min == 12
        assert config.track.bot_sort.t_miss_max == 60

    def test_reject_invalid_yaml_bot_sort_config(
        self, track_cli: Callable, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "user_config.otvision.yaml"
        config_file.write_text("TRACK:\n  BOT_SORT:\n    MATCH_THRESH: 1.1\n")

        with pytest.raises(InvalidOtvisionConfigError, match="MATCH_THRESH"):
            track_cli(argv=["--config", str(config_file), LOGFILE_OVERWRITE_CMD])

    @pytest.mark.parametrize(
        argnames="test_data",
        argvalues=[
            TEST_DATA_ALL_PARAMS_FROM_CLI_1,
            TEST_DATA_ALL_PARAMS_FROM_CLI_2,
            TEST_DATA_PARAMS_FROM_DEFAULT_CONFIG,
            TEST_DATA_PARAMS_FROM_CUSTOM_CONFIG,
        ],
    )
    @patch("track.TrackBuilder.update_current_config")
    @patch("track.TrackBuilder.build")
    def test_pass_track_cli(
        self,
        mock_build: Mock,
        mock_update_current_config: Mock,
        test_data: dict,
        track_cli: Callable,
    ) -> None:
        mock_otvision_track = Mock()
        mock_otvision_track.start = AsyncMock()
        mock_build.return_value = mock_otvision_track

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
        expected_config = create_expected_config_from_test_data(test_data)

        mock_update_current_config.update.assert_called_once_with(
            config=expected_config
        )
        mock_otvision_track.start.assert_awaited_once()

    @pytest.mark.parametrize(argnames="test_fail_data", argvalues=TEST_FAIL_DATA)
    @patch("track.TrackBuilder.build")
    def test_fail_wrong_types_passed_to_track_cli(
        self,
        mock_build: Mock,
        track_cli: Callable,
        capsys: pytest.CaptureFixture,
        test_fail_data: dict,
    ) -> None:
        with pytest.raises(SystemExit) as e:
            command = [*test_fail_data["passed"].split()]
            track_cli(argv=list(filter(None, command)))
        assert e.value.code == 2
        captured = capsys.readouterr()
        assert test_fail_data["error_msg_part"] in captured.err
        mock_build.assert_not_called()

    @pytest.mark.parametrize("passed", argvalues=["--config foo", "--paths foo"])
    @patch("track.TrackBuilder.build")
    def test_fail_not_existing_path_passed_to_track_cli(
        self, mock_build: Mock, track_cli: Callable, passed: str
    ) -> None:
        with pytest.raises(FileNotFoundError):
            command = [*passed.split(), LOGFILE_OVERWRITE_CMD]
            track_cli(argv=list(filter(None, command)))
        mock_build.assert_not_called()

    def test_fail_no_paths_passed_to_track_cli(self, track_cli: Callable) -> None:
        error_msg = (
            "No paths have been passed as command line args."
            + "No user config has been passed as command line arg."
        )
        with pytest.raises(CliParseError, match=error_msg):
            track_cli(argv=[LOGFILE_OVERWRITE_CMD])


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
        default_config = CONFIG_PARSER.parse(config_file_arg.split()[1])
    else:
        default_config = Config()

    # Map EXPECTED values to TrackConfig's relevant fields
    paths = test_data["paths"].get(EXPECTED, default_config.detect.paths)
    sigma_l = test_data["sigma_l"].get(EXPECTED, default_config.track.sigma_l)
    sigma_h = test_data["sigma_h"].get(EXPECTED, default_config.track.sigma_h)
    sigma_iou = test_data["sigma_iou"].get(EXPECTED, default_config.track.sigma_iou)
    t_min = test_data["t_min"].get(EXPECTED, default_config.track.t_min)
    t_miss_max = test_data["t_miss_max"].get(EXPECTED, default_config.track.t_miss_max)
    overwrite = test_data["overwrite"].get(EXPECTED, default_config.track.overwrite)
    paths = [str(Path(p).expanduser()) for p in paths]

    iou_config = _TrackIouConfig(
        sigma_l=sigma_l,
        sigma_h=sigma_h,
        sigma_iou=sigma_iou,
        t_min=t_min,
        t_miss_max=t_miss_max,
    )

    track_config = TrackConfig(
        paths=paths,
        run_chained=default_config.track.run_chained,
        iou=iou_config,
        overwrite=overwrite,
    )

    return Config(
        log=default_config.log,
        search_subdirs=default_config.search_subdirs,
        default_filetype=default_config.default_filetype,
        filetypes=default_config.filetypes,
        last_paths=default_config.last_paths,
        convert=default_config.convert,
        detect=default_config.detect,
        track=track_config,
        undistort=default_config.undistort,
        transform=default_config.transform,
        gui=default_config.gui,
    )
