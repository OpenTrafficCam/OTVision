import logging
from pathlib import Path

import yaml

from OTVision.domain.serialization import Deserializer
from OTVision.helpers.log import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)


class YamlDeserializer(Deserializer):
    def deserialize(self, file: Path) -> dict:
        with open(file, "r") as stream:
            try:
                yaml_config = yaml.safe_load(stream)
            except yaml.YAMLError:
                log.exception("Unable to parse user config. Using default config.")
                raise
        return yaml_config
