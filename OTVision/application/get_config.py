from pathlib import Path

from OTVision.config import Config, ConfigParser
from OTVision.domain.cli import CliArgs

DEFAULT_USER_CONFIG = "user_config.otvision.yaml"


class GetConfig:
    def __init__(self, config_parser: ConfigParser) -> None:
        self._config_parser = config_parser

    def get(self, args: CliArgs) -> Config:
        config = self._get_config(args)
        return config

    def _get_config(self, args: CliArgs) -> Config:
        if config_file := args.get_config_file():
            return self._config_parser.parse(config_file)
        else:
            user_config_cwd = Path.cwd() / DEFAULT_USER_CONFIG

            if user_config_cwd.exists():
                return self._config_parser.parse(user_config_cwd)
            return Config()
