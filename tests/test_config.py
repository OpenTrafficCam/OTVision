from pathlib import Path

import pytest

from OTVision import config


@pytest.fixture
def user_config(test_data_dir: Path) -> Path:
    return test_data_dir / "config" / "user_config.example.yaml"


class TestConfigParser:
    def test_parse_yaml(self, user_config: Path) -> None:
        config.parse_user_config(str(user_config))
