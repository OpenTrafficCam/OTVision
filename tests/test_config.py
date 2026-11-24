from pathlib import Path

import pytest

from OTVision import config
from OTVision.application.config import Config
from OTVision.plugin.yaml_serialization import YamlDeserializer


@pytest.fixture
def user_config(test_data_dir: Path) -> Path:
    return test_data_dir / "config" / "user_config.otvision.yaml"


@pytest.fixture
def json_file(test_data_dir: Path) -> Path:
    return test_data_dir / "config" / "example.json"


@pytest.fixture
def default_config() -> dict:
    return config.CONFIG.copy()


class TestConfig:
    def test_legacy_and_example_configs_include_tracker_type(self) -> None:
        example_path = Path(__file__).resolve().parents[1] / "user_config.otvision.yaml"
        example = YamlDeserializer().deserialize(example_path)

        for track in (config.CONFIG["TRACK"], example["TRACK"]):
            assert track["TRACKER_TYPE"] == "iou"
            keys = list(track)
            assert keys.index("TRACKER_TYPE") < keys.index("IOU")

    def test_legacy_and_example_configs_include_bot_sort_defaults(
        self, default_config: dict
    ) -> None:
        example_path = Path(__file__).resolve().parents[1] / "user_config.otvision.yaml"
        example = YamlDeserializer().deserialize(example_path)
        expected = Config().to_dict()["TRACK"]["BOT_SORT"]

        assert default_config["TRACK"]["BOT_SORT"] == expected
        assert example["TRACK"]["BOT_SORT"] == expected

    def test_default_config_from_parser_identical_to_dict(
        self, default_config: dict
    ) -> None:
        result = Config().to_dict()
        assert result == default_config

    def test_parse_user_config_overwrite_default(
        self, default_config: dict, user_config: Path
    ) -> None:
        assert config.CONFIG == default_config
        config.parse_user_config(str(user_config))
        assert config.CONFIG != default_config
        default_config["DETECT"]["YOLO"]["WEIGHTS"] = "my_weights"
        assert config.CONFIG == default_config
