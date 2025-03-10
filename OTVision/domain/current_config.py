from OTVision.config import Config


class CurrentConfig:
    def __init__(self, config: Config) -> None:
        self.__config = config

    def get(self) -> Config:
        return self.__config

    def update(self, config: Config) -> None:
        self.__config = config
