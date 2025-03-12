from OTVision.config import Config
from OTVision.domain.current_config import CurrentConfig


class GetCurrentConfig:
    def __init__(self, current_config: CurrentConfig) -> None:
        self._current_config = current_config

    def get(self) -> Config:
        return self._current_config.get()
