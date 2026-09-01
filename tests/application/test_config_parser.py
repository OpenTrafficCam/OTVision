from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from OTVision.application.config import (
    Config,
    DetectConfig,
    StreamConfig,
    TrackConfig,
    YoloConfig,
    _TrackBotSortConfig,
    _TrackIouConfig,
)
from OTVision.application.config_parser import ConfigParser, InvalidOtvisionConfigError
from OTVision.domain.tracker import TrackerLifecycle, TrackerType
from OTVision.plugin.ffmpeg_video_writer import (
    ConstantRateFactor,
    EncodingSpeed,
    VideoCodec,
)
from OTVision.plugin.yaml_serialization import YamlDeserializer

# Ensure paths are translated to the respective platform (unix, windows)
VIDEO_1 = str(Path("tests/data/video1.mp4"))
VIDEO_2 = str(Path("tests/data/video2.mp4"))


class TestConfigParser:
    @pytest.fixture
    def given_deserializer(self) -> Mock:
        return Mock()

    @pytest.fixture
    def given_config_parser(self, given_deserializer: Mock) -> ConfigParser:
        return ConfigParser(given_deserializer)

    def test_parse_detect_config_with_fully_specified_config(
        self, given_config_parser: ConfigParser
    ) -> None:
        detect_dict = {
            "PATHS": [VIDEO_1, VIDEO_2],
            "RUN_CHAINED": False,
            "YOLO": {
                "WEIGHTS": "yolov8m.pt",
                "CONF": 0.35,
                "IOU": 0.5,
                "IMGSIZE": 1280,
                "NORMALIZED": False,
            },
            "EXPECTED_DURATION": 3600,
            "OVERWRITE": False,
            "HALF_PRECISION": True,
            "START_TIME": "2025-10-15_14-30-00",
            "DETECT_START": 10,
            "DETECT_END": 100,
            "WRITE_VIDEO": True,
            "VIDEO_CODEC": "h264_nvenc",
            "ENCODING_SPEED": "medium",
            "CRF": "HIGH_QUALITY",
        }

        result = given_config_parser.parse_detect_config(detect_dict)

        expected = DetectConfig(
            paths=[VIDEO_1, VIDEO_2],
            run_chained=False,
            yolo_config=YoloConfig(
                weights="yolov8m.pt",
                conf=0.35,
                iou=0.5,
                img_size=1280,
                normalized=False,
            ),
            expected_duration=timedelta(seconds=3600),
            overwrite=False,
            half_precision=True,
            start_time=datetime(2025, 10, 15, 14, 30, 0),
            detect_start=10,
            detect_end=100,
            write_video=True,
            video_codec=VideoCodec.H264_NVENC,
            encoding_speed=EncodingSpeed.MEDIUM,
            crf=ConstantRateFactor.HIGH_QUALITY,
        )
        assert result == expected

    def test_parse_detect_config_with_empty_config(
        self, given_config_parser: ConfigParser
    ) -> None:
        detect_dict: dict = {}

        result = given_config_parser.parse_detect_config(detect_dict)

        expected = DetectConfig()
        assert result == expected


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
        return TrackConfig(iou=iou_config, tracker_type=TrackerType.IOU)

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


class TestTrackerTypeParsing:
    """Parsing and validation of tracker selection and BoT-SORT params."""

    def test_parses_uppercase_botsort_keys_from_yaml(self) -> None:
        """Shipped YAML uses UPPERCASE keys; they normalize to Ultralytics names."""
        target = ConfigParser(YamlDeserializer())

        config = target.parse_track_botsort_config(
            {"T_MIN": 5, "T_MISS_MAX": 60, "TRACK_HIGH_THRESH": 0.3}
        )

        assert config.t_min == 5
        assert config.t_miss_max == 60
        assert config.tracker_params["track_high_thresh"] == 0.3

    def test_rejects_unknown_botsort_param(self) -> None:
        """A typo'd param fails at parse time, not deep inside Ultralytics."""
        target = ConfigParser(YamlDeserializer())

        with pytest.raises(InvalidOtvisionConfigError, match="track_high_tresh"):
            target.parse_track_botsort_config({"TRACK_HIGH_TRESH": 0.3})

    def test_rejects_nested_ultralytics_tracker_type(self) -> None:
        """The inert nested TRACKER_TYPE is rejected to avoid confusion."""
        target = ConfigParser(YamlDeserializer())

        with pytest.raises(InvalidOtvisionConfigError, match="not supported"):
            target.parse_track_botsort_config({"TRACKER_TYPE": "bytetrack"})

    def test_track_buffer_is_an_accepted_override(self) -> None:
        """TRACK_BUFFER is absent from defaults but remains a valid override."""
        target = ConfigParser(YamlDeserializer())

        config = target.parse_track_botsort_config({"TRACK_BUFFER": 90})

        assert config.tracker_params["track_buffer"] == 90

    def test_rejects_unknown_tracker_type(self) -> None:
        """An unknown TRACK.TRACKER_TYPE is rejected with the valid options."""
        target = ConfigParser(YamlDeserializer())

        with pytest.raises(InvalidOtvisionConfigError, match="Unknown TRACK.TRACKER"):
            target.parse_tracker_type("deepsort")

    def test_tracker_type_is_case_insensitive(self) -> None:
        """Tracker selection tolerates casing differences in YAML."""
        target = ConfigParser(YamlDeserializer())

        assert target.parse_tracker_type("BotSort") is TrackerType.BOTSORT

    def test_absent_tracker_type_defaults_to_iou(self) -> None:
        """IOU stays the default tracker when YAML omits the selector."""
        target = ConfigParser(YamlDeserializer())

        assert target.parse_tracker_type(None) is TrackerType.IOU


class TestTrackConfigTrackerTypeCoercion:
    """`TrackConfig` normalizes tracker_type however it was constructed."""

    def test_plain_string_is_coerced_to_the_enum(self) -> None:
        """A raw string must not silently take the wrong dispatch branch.

        StrEnum members compare equal to their string value but are not
        identical, so an un-coerced `"botsort"` would fail every `is` check
        and select IOU's lifecycle instead.
        """
        config = TrackConfig(
            iou=_TrackIouConfig(
                sigma_l=0.27, sigma_h=0.42, sigma_iou=0.38, t_min=5, t_miss_max=51
            ),
            botsort=_TrackBotSortConfig(t_min=7, t_miss_max=60),
            tracker_type="botsort",  # type: ignore[arg-type]
        )

        assert config.tracker_type is TrackerType.BOTSORT
        assert config.lifecycle == TrackerLifecycle(t_min=7, t_miss_max=60)

    def test_unknown_string_is_rejected(self) -> None:
        """An unknown tracker name fails at construction."""
        with pytest.raises(ValueError, match="not a valid TrackerType"):
            TrackConfig(tracker_type="deepsort")  # type: ignore[arg-type]
