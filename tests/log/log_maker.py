import logging

from OTVision.helpers.log import LOG_LEVEL_INTEGERS, LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)


class MyCustomError(Exception):
    "This is a custom error message"


class LogMaker:
    def log_str_level(self, level: str) -> None:
        log_msg = f"This is a {level} log"
        match level:
            case "DEBUG":
                log.debug(log_msg)
            case "INFO":
                log.info(log_msg)
            case "WARNING":
                log.warning(log_msg)
            case "ERROR":
                log.error(log_msg)
            case "CRITICAL":
                log.critical(log_msg)

    def log_int_level(self, level: str) -> None:
        level_int = LOG_LEVEL_INTEGERS[level]
        log_msg = f"This is a numeric level {level_int} a.k.a. {level} log"
        match level_int:
            case 10:
                log.debug(log_msg)
            case 20:
                log.info(log_msg)
            case 30:
                log.warning(log_msg)
            case 40:
                log.error(log_msg)
            case 50:
                log.critical(log_msg)

    def raise_error_and_log(self, log_msg: str) -> None:
        try:
            raise MyCustomError
        except MyCustomError:
            log.exception(msg=log_msg)
