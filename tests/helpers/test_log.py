import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable
from unittest.mock import Mock, call, patch

import pytest

from OTVision.helpers.log import LOG_EXT, LogFileAlreadyExists, _OTVisionLogger
from tests.conftest import YieldFixture

DATETIME_FORMAT = r"%Y-%m-%d_%H-%M-%S"
FIRST_DATETIME = datetime(2020, 1, 1, 14, 0, 0)
SECOND_DATETIME = datetime(2020, 1, 1, 14, 15, 0)
FIRST_DATETIME_STR = FIRST_DATETIME.strftime(DATETIME_FORMAT)
SECOND_DATETIME_STR = SECOND_DATETIME.strftime(DATETIME_FORMAT)
EXPECTED_LOG_LEVEL = "DEBUG"


@pytest.fixture
def logfile_arg_existing_dir(test_data_tmp_dir: Path) -> YieldFixture[Path]:
    arg = test_data_tmp_dir / "my_new_project"
    arg.mkdir()
    yield arg
    shutil.rmtree(arg)


@pytest.fixture
def logfile_arg_non_existing_dir(test_data_tmp_dir: Path) -> YieldFixture[Path]:
    arg = test_data_tmp_dir / "non_existing_dir"
    yield arg
    shutil.rmtree(arg)


@pytest.fixture
def logfile_arg_exists_true(test_data_tmp_dir: Path) -> YieldFixture[Path]:
    arg = test_data_tmp_dir / "my_logfile.log"
    arg.touch()
    yield arg
    arg.unlink()


@pytest.fixture
def logfile_arg_exists_false(test_data_tmp_dir: Path) -> YieldFixture[Path]:
    arg = test_data_tmp_dir / "my_new_logfile_.log"
    yield arg
    arg.unlink()


class TestOTVisionLogger:
    @patch("OTVision.helpers.log._OTVisionLogger._add_handler")
    @patch("OTVision.helpers.log.logging")
    def test_init_with_logfile_being_an_existing_dir(
        self, mock_logging: Mock, mock_add_handler: Mock, logfile_arg_existing_dir: Path
    ) -> None:
        given_datetime_provider = create_datetime_provider([FIRST_DATETIME])
        target = _OTVisionLogger(datetime_provider=given_datetime_provider)

        assert logfile_arg_existing_dir.exists()
        target.add_file_handler(log_file=logfile_arg_existing_dir)

        expected_logfile = create_expected_log_file(
            logfile_arg_existing_dir, FIRST_DATETIME_STR
        )
        assert logfile_arg_existing_dir.exists()
        assert expected_logfile.exists()
        mock_add_handler.assert_called_once_with(
            mock_logging.FileHandler(), EXPECTED_LOG_LEVEL
        )

    @patch("OTVision.helpers.log._OTVisionLogger._add_handler")
    @patch("OTVision.helpers.log.logging")
    def test_init_with_logfile_being_a_non_existing_dir(
        self,
        mock_logging: Mock,
        mock_add_handler: Mock,
        logfile_arg_non_existing_dir: Path,
    ) -> None:
        given_datetime_provider = create_datetime_provider(
            [FIRST_DATETIME, SECOND_DATETIME]
        )
        target = _OTVisionLogger(datetime_provider=given_datetime_provider)

        assert not logfile_arg_non_existing_dir.exists()
        target.add_file_handler(log_file=logfile_arg_non_existing_dir)
        target.add_file_handler(log_file=logfile_arg_non_existing_dir)

        expected_first_logfile = create_expected_log_file(
            logfile_arg_non_existing_dir, FIRST_DATETIME_STR
        )
        expected_second_logfile = create_expected_log_file(
            logfile_arg_non_existing_dir, SECOND_DATETIME_STR
        )

        assert logfile_arg_non_existing_dir.exists()
        assert expected_first_logfile.exists()
        assert expected_second_logfile.exists()
        assert mock_add_handler.call_args_list == [
            call(mock_logging.FileHandler(), EXPECTED_LOG_LEVEL),
            call(mock_logging.FileHandler(), EXPECTED_LOG_LEVEL),
        ]

    @patch("OTVision.helpers.log._OTVisionLogger._add_handler")
    @patch("OTVision.helpers.log.logging")
    def test_init_with_non_existing_logfile(
        self,
        mock_logging: Mock,
        mock_add_handler: Mock,
        logfile_arg_exists_false: Path,
    ) -> None:
        given_datetime_provider = create_datetime_provider([FIRST_DATETIME])
        target = _OTVisionLogger(datetime_provider=given_datetime_provider)

        assert not logfile_arg_exists_false.exists()
        target.add_file_handler(log_file=logfile_arg_exists_false)

        expected_first_logfile = create_expected_log_file(
            logfile_arg_exists_false, FIRST_DATETIME_STR
        )

        assert expected_first_logfile.exists()
        assert mock_add_handler.call_args_list == [
            call(mock_logging.FileHandler(), EXPECTED_LOG_LEVEL),
        ]

    @patch("OTVision.helpers.log._OTVisionLogger._add_handler")
    @patch("OTVision.helpers.log.logging")
    def test_init_with_existing_logfile_overwrite_true(
        self,
        mock_logging: Mock,
        mock_add_handler: Mock,
        logfile_arg_exists_true: Path,
    ) -> None:
        given_datetime_provider = create_datetime_provider([FIRST_DATETIME])
        target = _OTVisionLogger(datetime_provider=given_datetime_provider)

        assert logfile_arg_exists_true.exists()
        target.add_file_handler(log_file=logfile_arg_exists_true, overwrite=True)
        expected_first_logfile = create_expected_log_file(
            logfile_arg_exists_true, FIRST_DATETIME_STR
        )
        assert expected_first_logfile.exists()
        assert mock_add_handler.call_args_list == [
            call(mock_logging.FileHandler(), EXPECTED_LOG_LEVEL),
        ]

    @patch("OTVision.helpers.log._OTVisionLogger._add_handler")
    @patch("OTVision.helpers.log.logging")
    def test_init_with_existing_logfile_overwrite_false(
        self,
        mock_logging: Mock,
        mock_add_handler: Mock,
        logfile_arg_exists_true: Path,
    ) -> None:
        given_datetime_provider = create_datetime_provider([FIRST_DATETIME])
        target = _OTVisionLogger(datetime_provider=given_datetime_provider)

        assert logfile_arg_exists_true.exists()
        with pytest.raises(LogFileAlreadyExists):
            target.add_file_handler(log_file=logfile_arg_exists_true, overwrite=False)

        mock_add_handler.assert_not_called()


def create_datetime_provider(return_values: list[datetime]) -> Callable[[], datetime]:
    mock = Mock()
    mock.side_effect = return_values
    return mock


def create_expected_log_file(given_log_file: Path, expected_datetime: str) -> Path:
    if given_log_file.suffix == f".{LOG_EXT}":
        return given_log_file

    return given_log_file / f"logs/{expected_datetime}.log"
