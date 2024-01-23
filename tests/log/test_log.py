import logging
from filecmp import cmp
from pathlib import Path

import pytest

from OTVision.helpers.log import (
    LOG_LEVEL_INTEGERS,
    VALID_LOG_LEVELS,
    LogFileAlreadyExists,
    log,
)
from tests.conftest import YieldFixture

from .log_maker import LogMaker


class WrongNumberOfFilesFoundError(Exception):
    "Too few or too many log files have been created during this test run"


@pytest.fixture()
def teardown_handlers_after_test() -> YieldFixture:
    yield
    log._remove_handlers()


@pytest.mark.usefixtures("teardown_handlers_after_test")
class TestLog:
    log_maker: LogMaker = LogMaker()

    log.formatter = logging.Formatter(
        "%(levelname)s (%(filename)s::%(funcName)s" "::%(lineno)d): %(message)s"
    )

    @pytest.mark.parametrize("level", VALID_LOG_LEVELS)
    def test_logger_logs_correct_message_for_level_in_other_file(
        self, level: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        self.log_maker.log_message_on_str_level(level=level)

        assert f"This is a {level} log" in caplog.text

    def test_logger_logs_caught_exception_properly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        log_msg = "Caught exception"

        self.log_maker.log_message_on_caught_error(log_msg)

        assert log_msg in caplog.text
        assert "CaughtError" in caplog.text

    @pytest.mark.parametrize("logger_level", VALID_LOG_LEVELS)
    @pytest.mark.parametrize("log_level", VALID_LOG_LEVELS)
    def test_console_handler_logs_correct_message_only_above_his_level(
        self, logger_level: str, log_level: str, capsys: pytest.CaptureFixture
    ) -> None:
        log_level_int = LOG_LEVEL_INTEGERS[log_level]
        logger_level_int = LOG_LEVEL_INTEGERS[logger_level]

        log.add_console_handler(level=logger_level)

        self.log_maker.log_message_on_int_level(level=log_level)

        stdout, stderr = capsys.readouterr()

        # Check if log message including level is in console output
        if log_level_int >= logger_level_int:
            assert (
                f"This is a numeric level {log_level_int} a.k.a. {log_level} log"
                in stdout
            )
        else:
            assert stdout == ""

    def test_file_handler_logs_correct_content_to_file(
        self,
        test_data_tmp_dir: Path,
        test_data_dir: Path,
    ) -> None:
        log_file = test_data_tmp_dir / "log/_otvision_logs/test.log"
        log.add_file_handler(level="DEBUG", log_file=log_file)

        for level in VALID_LOG_LEVELS:
            self.log_maker.log_message_on_str_level(level=level)

        ref_log_file = test_data_dir / "log/_otvision_logs/test.log"

        assert log_file.exists()
        assert cmp(ref_log_file, log_file)

    def test_write_to_existing_log_file_fails_without_required_flag(
        self, test_data_tmp_dir: Path
    ) -> None:
        log_file = test_data_tmp_dir / "my_log_to_overwrite.log"
        log_file.touch()
        with pytest.raises(LogFileAlreadyExists):
            log.add_file_handler(log_file, overwrite=False)

    def test_overwrite_log_with_with_required_flag(
        self, test_data_tmp_dir: Path
    ) -> None:
        log_file = test_data_tmp_dir / "my_log_to_overwrite.log"
        log_file.touch()
        assert log_file.exists()
        log.add_file_handler(log_file, overwrite=True)
