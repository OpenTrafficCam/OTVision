from OTVision.config import Config
from OTVision.domain.current_config import CurrentConfig


class UpdateCurrentConfig:
    def __init__(self, current_config: CurrentConfig) -> None:
        self._current_config = current_config

    def update(self, config: Config) -> None:
        self._current_config.update(config)
