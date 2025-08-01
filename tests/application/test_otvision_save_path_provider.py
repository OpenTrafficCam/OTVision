from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from OTVision.application.config import Config
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.application.otvision_save_path_provider import (
    OtvisionSavePathProvider,
    derive_filename,
)

CONFIG = Config()

OTDET_FILE_TYPE = ".otdet"


class TestOtvisionSavePathProvider:
    @patch("OTVision.application.otvision_save_path_provider.derive_filename")
    def test_provide(self, mock_derive_filename: Mock) -> None:
        expected_save_path = Mock()
        mock_derive_filename.return_value = expected_save_path
        given_get_current_config = self.create_get_current_config()
        given_video = "video.mp4"
        target = OtvisionSavePathProvider(given_get_current_config)

        actual = target.provide(given_video, OTDET_FILE_TYPE)

        assert actual == expected_save_path
        given_get_current_config.get.assert_called_once()
        mock_derive_filename.assert_called_once_with(
            video_file=Path(given_video),
            file_type=OTDET_FILE_TYPE,
            detect_start=CONFIG.detect.detect_start,
            detect_end=CONFIG.detect.detect_end,
        )

    def create_get_current_config(self) -> Mock:
        mock = Mock(spec=GetCurrentConfig)
        mock.get.return_value = CONFIG
        return mock


class TestDeriveFilename:

    @pytest.mark.parametrize(
        "video_file, detection_file, detect_start, detect_end",
        [
            ("video.mp4", "video.otdet", None, None),
            ("video.mp4", "video_end_20.otdet", None, 20),
            ("video.mp4", "video_start_10.otdet", 10, None),
            ("video.mp4", "video_start_10_end_20.otdet", 10, 20),
        ],
    )
    def test_derive_filename(
        self,
        video_file: str,
        detection_file: str,
        detect_start: int | None,
        detect_end: int | None,
    ) -> None:
        actual = derive_filename(
            video_file=Path(video_file),
            detect_start=detect_start,
            detect_end=detect_end,
            file_type=".otdet",
        )

        assert actual == Path(detection_file)
