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


class TestBotSortConfig:
    @pytest.mark.parametrize(
        "track", [{}, {"BOT_SORT": {}}, {"BOT_SORT": None}, {"IOU": {"T_MIN": 2}}]
    )
    def test_missing_block_or_keys_use_defaults(self, track: dict) -> None:
        parser = ConfigParser(YamlDeserializer())

        config = parser.parse_from_dict({"TRACK": track})

        assert config.track.bot_sort == _TrackBotSortConfig()

    def test_parse_yaml_and_round_trip_config(self, tmp_path: Path) -> None:
        parser = ConfigParser(YamlDeserializer())
        config_file = tmp_path / "user_config.otvision.yaml"
        # Every value differs from its default, and the frame counts and
        # thresholds sit at the edges of their valid ranges.
        config_file.write_text(
            "TRACK:\n"
            "  BOT_SORT:\n"
            "    T_MIN: 0\n"
            "    T_MISS_MAX: 20\n"
            "    TRACK_HIGH_THRESH: 1\n"
            "    TRACK_LOW_THRESH: 0\n"
            "    NEW_TRACK_THRESH: 0.4\n"
            "    MATCH_THRESH: 0.7\n"
            "    FUSE_SCORE: false\n"
        )

        config = parser.parse(config_file)

        assert config.track.bot_sort == _TrackBotSortConfig(
            t_min=0,
            t_miss_max=20,
            track_high_thresh=1,
            track_low_thresh=0,
            new_track_thresh=0.4,
            match_thresh=0.7,
            fuse_score=False,
        )
        assert parser.parse_from_dict(config.to_dict()) == config

    @pytest.mark.parametrize("key", ["TRACK_BUFFER", "WITH_REID", "GMC_METHOD"])
    def test_reject_unknown_bot_sort_key(self, key: str) -> None:
        parser = ConfigParser(YamlDeserializer())

        with pytest.raises(InvalidOtvisionConfigError, match=key):
            parser.parse_from_dict({"TRACK": {"BOT_SORT": {key: False}}})

    @pytest.mark.parametrize("block", [False, [], "settings"])
    def test_reject_bot_sort_block_that_is_not_a_mapping(self, block: object) -> None:
        parser = ConfigParser(YamlDeserializer())

        with pytest.raises(InvalidOtvisionConfigError, match="BOT_SORT"):
            parser.parse_from_dict({"TRACK": {"BOT_SORT": block}})

    def test_ignore_unknown_config_keys_outside_bot_sort(self) -> None:
        parser = ConfigParser(YamlDeserializer())

        config = parser.parse_from_dict(
            {"UNKNOWN": 1, "TRACK": {"UNKNOWN": 2, "IOU": {"UNKNOWN": 3}}}
        )

        assert config == Config()

    @pytest.mark.parametrize(
        "key",
        ["TRACK_HIGH_THRESH", "TRACK_LOW_THRESH", "NEW_TRACK_THRESH", "MATCH_THRESH"],
    )
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(-0.1, id="below-0"),
            pytest.param(1.1, id="above-1"),
            pytest.param(float("nan"), id="nan"),
            pytest.param("0.5", id="string"),
            pytest.param(True, id="bool"),
        ],
    )
    def test_reject_invalid_bot_sort_threshold(self, key: str, value: object) -> None:
        parser = ConfigParser(YamlDeserializer())

        with pytest.raises(InvalidOtvisionConfigError, match=key):
            parser.parse_from_dict({"TRACK": {"BOT_SORT": {key: value}}})

    @pytest.mark.parametrize(
        "low_threshold",
        [
            pytest.param(_TrackBotSortConfig.track_high_thresh, id="equal"),
            pytest.param(_TrackBotSortConfig.track_high_thresh + 0.1, id="above"),
        ],
    )
    def test_reject_bot_sort_low_threshold_at_or_above_high(
        self, low_threshold: float
    ) -> None:
        parser = ConfigParser(YamlDeserializer())

        with pytest.raises(
            InvalidOtvisionConfigError, match="TRACK_LOW_THRESH.*TRACK_HIGH_THRESH"
        ):
            parser.parse_from_dict(
                {"TRACK": {"BOT_SORT": {"TRACK_LOW_THRESH": low_threshold}}}
            )

    @pytest.mark.parametrize("key", ["T_MIN", "T_MISS_MAX"])
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(-1, id="negative"),
            pytest.param(0.5, id="fraction"),
            pytest.param(1.0, id="whole-float"),
            pytest.param("3", id="string"),
            pytest.param(True, id="bool"),
        ],
    )
    def test_reject_invalid_bot_sort_frame_count(self, key: str, value: object) -> None:
        parser = ConfigParser(YamlDeserializer())

        with pytest.raises(InvalidOtvisionConfigError, match=key):
            parser.parse_from_dict({"TRACK": {"BOT_SORT": {key: value}}})

    @pytest.mark.parametrize("value", [0, "false", None])
    def test_reject_non_boolean_bot_sort_fuse_score(self, value: object) -> None:
        parser = ConfigParser(YamlDeserializer())

        with pytest.raises(InvalidOtvisionConfigError, match="FUSE_SCORE"):
            parser.parse_from_dict({"TRACK": {"BOT_SORT": {"FUSE_SCORE": value}}})


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
