from pathlib import Path
from unittest.mock import Mock

import pytest

from OTVision.application.config import (
    Config,
    StreamConfig,
    TrackConfig,
    _TrackIouConfig,
)
from OTVision.application.config_parser import ConfigParser, InvalidOtvisionConfigError


class TestConfigParserValidateFlushBufferSupportTrackLifecycle:
    """Test suite for validate_flush_buffer_support_track_lifecycle method.

    This test suite validates the flush buffer size configuration against track
    lifecycle parameters to ensure tracks can complete their full lifecycle
    before being flushed in streaming mode.
    """

    def test_validate_with_no_stream_config_passes(
        self, given_config_parser: ConfigParser
    ) -> None:
        given_config = self._build_config(stream_config=None)

        # When: Validating flush buffer support for track lifecycle
        # Then: No exception should be raised
        given_config_parser.validate_flush_buffer_support_track_lifecycle(given_config)

    @pytest.mark.parametrize(
        "t_min, t_miss_max, flush_buffer_size, should_raise_error",
        [
            (5, 51, 100, False),  # t_min, t_miss_max < flush_buffer_size
            (5, 51, 51, True),  # t_miss_max == flush_buffer_size
            (5, 51, 50, True),  # t_miss_max > flush_buffer_size
            (5, 51, 5, True),  # t_min == flush_buffer_size
            (5, 51, 4, True),  # t_min > flush_buffer_size
        ],
    )
    def test_validate_with_various_track_config_values(
        self,
        given_config_parser: ConfigParser,
        t_min: int,
        t_miss_max: int,
        flush_buffer_size: int,
        should_raise_error: bool,
    ) -> None:
        # Given: A configuration with the specified parameters
        given_track_config = self._build_track_config(
            t_min=t_min, t_miss_max=t_miss_max
        )
        given_stream_config = self._build_stream_config(
            flush_buffer_size=flush_buffer_size
        )
        given_config = self._build_config(
            stream_config=given_stream_config, track_config=given_track_config
        )

        if should_raise_error:
            with pytest.raises(InvalidOtvisionConfigError):
                given_config_parser.validate_flush_buffer_support_track_lifecycle(
                    given_config
                )
        else:
            # When: Validating flush buffer support for track lifecycle
            # Then: No exception should be raised
            given_config_parser.validate_flush_buffer_support_track_lifecycle(
                given_config
            )

    @pytest.fixture
    def given_deserializer(self) -> Mock:
        return Mock()

    @pytest.fixture
    def given_config_parser(self, given_deserializer: Mock) -> ConfigParser:
        return ConfigParser(given_deserializer)

    def _build_stream_config(
        self,
        name: str = "test_stream",
        source: str = "rtsp://example.com",
        save_dir: Path = Path("/tmp"),
        flush_buffer_size: int = 100,
    ) -> StreamConfig:
        """Build a StreamConfig instance for testing.

        Args:
            name: Stream name.
            source: Stream source URL.
            save_dir: Directory to save stream data.
            flush_buffer_size: Size of the flush buffer.

        Returns:
            StreamConfig: Configured stream instance.
        """
        return StreamConfig(
            name=name,
            source=source,
            save_dir=save_dir,
            flush_buffer_size=flush_buffer_size,
        )

    def _build_track_config(self, t_min: int = 5, t_miss_max: int = 51) -> TrackConfig:
        """Build a TrackConfig instance for testing.

        Args:
            t_min: Minimum track length.
            t_miss_max: Maximum missing frames before track termination.

        Returns:
            TrackConfig: Configured track instance.
        """
        iou_config = _TrackIouConfig(t_min=t_min, t_miss_max=t_miss_max)
        return TrackConfig(iou=iou_config)

    def _build_config(
        self,
        stream_config: StreamConfig | None = None,
        track_config: TrackConfig | None = None,
    ) -> Config:
        """Build a Config instance for testing.

        Args:
            stream_config: Optional stream configuration.
            track_config: Optional track configuration.

        Returns:
            Config: Configured application instance.
        """
        return Config(
            stream=stream_config,
            track=track_config or TrackConfig(),
        )
