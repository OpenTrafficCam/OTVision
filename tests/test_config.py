from pathlib import Path

import pytest

from OTVision import config


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
    def test_default_config_from_parser_identical_to_dict(
        self, default_config: dict
    ) -> None:
        result = config.Config().to_dict()
        assert result == default_config

    def test_parse_user_config_overwrite_default(
        self, default_config: dict, user_config: Path
    ) -> None:
        assert config.CONFIG == default_config
        config.parse_user_config(str(user_config))
        assert config.CONFIG != default_config
        default_config["DETECT"]["YOLO"]["WEIGHTS"] = "my_weights"
        assert config.CONFIG == default_config
