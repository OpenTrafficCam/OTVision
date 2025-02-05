import logging
from logging import Logger
from pathlib import Path

from OTVision.config import Config
from OTVision.helpers.log import LOGGER_NAME, log


def logger() -> Logger:
    return logging.getLogger(LOGGER_NAME)


class ConfigureLogger:
    def configure(
        self, config: Config, log_file: Path, logfile_overwrite: bool
    ) -> Logger:
        log.add_console_handler(level=config.log.log_level_console)
        log.add_file_handler(
            log_file.expanduser(),
            config.log.log_level_file,
            logfile_overwrite,
        )
        return logging.getLogger(LOGGER_NAME)
