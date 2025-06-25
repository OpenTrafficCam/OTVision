from datetime import datetime, timedelta
from pathlib import Path

import pytest

from OTVision import dataformat, version
from OTVision.dataformat import CLASS, CONFIDENCE, H, W, X, Y
from OTVision.detect.otdet import (
    OtdetBuilder,
    OtdetBuilderConfig,
    OtdetBuilderError,
    parse_video_length,
    serialize_video_length,
)
from OTVision.domain.detection import Detection
from OTVision.domain.frame import DetectedFrame

GIVEN_ACTUAL_DURATION = timedelta(
    hours=36, seconds=33, milliseconds=12, microseconds=15
)
EXPECTED_VIDEO_LENGTH = "36:00:33"
DETECTED_FRAME_OUTPUT = "path/to/output.mp4"


def create_expected_video_metadata(
    config: OtdetBuilderConfig, number_of_frames: int
) -> dict:
    """Create the expected video metadata based on the given config."""
    video_metadata = {
        dataformat.FILENAME: str(Path(config.source).stem),
        dataformat.FILETYPE: str(Path(config.source).suffix),
        dataformat.WIDTH: config.video_width,
        dataformat.HEIGHT: config.video_height,
        dataformat.RECORDED_FPS: config.recorded_fps,
        dataformat.ACTUAL_FPS: config.actual_fps,
        dataformat.NUMBER_OF_FRAMES: number_of_frames,
        dataformat.RECORDED_START_DATE: config.recorded_start_date.timestamp(),
        dataformat.LENGTH: EXPECTED_VIDEO_LENGTH,
    }
    if config.expected_duration:
        video_metadata[dataformat.EXPECTED_DURATION] = int(
            config.expected_duration.total_seconds()
        )
    return video_metadata


def create_expected_detection_metadata(config: OtdetBuilderConfig) -> dict:
    """Create the expected detection metadata based on the given config."""
    return {
        dataformat.OTVISION_VERSION: version.otvision_version(),
        dataformat.MODEL: {
            dataformat.NAME: "YOLOv8",
            dataformat.WEIGHTS: str(config.detection_model),
            dataformat.IOU_THRESHOLD: config.iou,
            dataformat.IMAGE_SIZE: config.detection_img_size,
            dataformat.MAX_CONFIDENCE: config.conf,
            dataformat.HALF_PRECISION: config.half_precision,
            dataformat.CLASSES: config.classifications,
        },
        dataformat.CHUNKSIZE: config.chunksize,
        dataformat.NORMALIZED_BBOX: config.normalized,
        dataformat.DETECT_START: config.detect_start,
        dataformat.DETECT_END: config.detect_end,
    }


def create_expected_metadata(config: OtdetBuilderConfig, number_of_frames: int) -> dict:
    """Create the full expected metadata based on the given config."""
    return {
        dataformat.OTDET_VERSION: version.otdet_version(),
        dataformat.VIDEO: create_expected_video_metadata(config, number_of_frames),
        dataformat.DETECTION: create_expected_detection_metadata(config),
    }


def create_expected_data(
    frames: list[DetectedFrame],
) -> dict:
    """Create the expected data dictionary based on input detections."""
    data = {}
    for frame in frames:
        data[str(frame.no)] = {
            dataformat.DETECTIONS: [
                create_expected_detection(d) for d in frame.detections
            ],
            dataformat.OCCURRENCE: frame.occurrence.timestamp(),
        }
    return data


def create_expected_detection(detection: Detection) -> dict:
    return {
        CLASS: detection.label,
        CONFIDENCE: detection.conf,
        X: detection.x,
        Y: detection.y,
        W: detection.w,
        H: detection.h,
    }


def create_detected_frame(source: str, frame_number: int) -> DetectedFrame:
    detection = Detection(
        label="person",
        conf=0.9,
        x=100,
        y=200,
        w=100,
        h=200,
    )
    return DetectedFrame(
        source=source,
        output=DETECTED_FRAME_OUTPUT,
        no=frame_number,
        detections=[detection],
        occurrence=datetime(2020, 1, 1, 12, 0, 10),
    )


class TestOtdetBuilder:
    @pytest.fixture
    def builder(self) -> OtdetBuilder:
        """Fixture to provide a new instance of OtdetBuilder for every test."""
        return OtdetBuilder()

    @pytest.fixture
    def config(self) -> OtdetBuilderConfig:
        """Fixture to provide a predefined valid OtdetBuilderConfig."""
        return OtdetBuilderConfig(
            conf=0.5,
            iou=0.4,
            source="video.mp4",
            video_width=1920,
            video_height=1080,
            expected_duration=timedelta(seconds=300),
            actual_duration=GIVEN_ACTUAL_DURATION,
            recorded_fps=30.0,
            recorded_start_date=datetime(2020, 1, 1, 12, 0, 0),
            actual_fps=29.97,
            actual_frames=1000,
            detection_img_size=640,
            normalized=True,
            detection_model=Path("model.pt"),
            half_precision=False,
            chunksize=32,
            classifications={0: "person", 1: "car"},
            detect_start=300,
            detect_end=600,
        )

    def test_builder_without_config_raises_error(self, builder: OtdetBuilder) -> None:
        """Test that accessing config without setting it raises an error.

        #Requirement https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7188

        """  # noqa
        with pytest.raises(OtdetBuilderError, match="Otdet builder config is not set"):
            _ = builder.config

    def test_builder_add_config_successfully(
        self, builder: OtdetBuilder, config: OtdetBuilderConfig
    ) -> None:
        """Test that configuration can be added to the builder.

        #Requirement https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7188

        """  # noqa
        actual = builder.add_config(config)
        assert builder.config == config
        assert actual == builder

    def test_build_metadata(
        self, builder: OtdetBuilder, config: OtdetBuilderConfig
    ) -> None:
        """Test the metadata generation.

        https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7188

        """  # noqa
        builder.add_config(config)
        actual = builder._build_metadata(number_of_frames=10)

        expected = create_expected_metadata(config, number_of_frames=10)
        assert actual == expected

    def test_build_data(
        self,
        builder: OtdetBuilder,
        config: OtdetBuilderConfig,
    ) -> None:
        """Test the data generation with detection objects.

        https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7188

        """  # noqa
        builder.add_config(config)
        source = config.source

        given: list[DetectedFrame] = [
            create_detected_frame(source, 1),
            create_detected_frame(source, 2),
            create_detected_frame(source, 3),
        ]
        actual = builder._build_data(given)

        expected = create_expected_data(given)
        assert actual == expected

    def test_build_full_result(
        self,
        builder: OtdetBuilder,
        config: OtdetBuilderConfig,
    ) -> None:
        """Test the full build method.


        https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7188
        """  # noqa
        builder.add_config(config)
        source = config.source

        given: list[DetectedFrame] = [
            create_detected_frame(source, 1),
            create_detected_frame(source, 2),
            create_detected_frame(source, 3),
        ]
        actual = builder.build(given)

        expected_metadata = create_expected_metadata(
            config, number_of_frames=len(given)
        )
        expected_data = create_expected_data(given)
        expected = {
            dataformat.METADATA: expected_metadata,
            dataformat.DATA: expected_data,
        }
        assert actual == expected

    def test_reset_builder(
        self, builder: OtdetBuilder, config: OtdetBuilderConfig
    ) -> None:
        """Test that the builder is reset after building.

        https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7188

        """  # noqa
        builder.add_config(config)
        assert builder.config == config

        actual = builder.reset()
        assert actual == builder
        assert builder._config is None

    def test_empty_detections_builds_valid_data(
        self, builder: OtdetBuilder, config: OtdetBuilderConfig
    ) -> None:
        """Test that an empty detection list builds valid data.

        https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7188

        """  # noqa
        builder.add_config(config)
        actual = builder.build([])

        expected = {
            dataformat.METADATA: create_expected_metadata(config, 0),
            dataformat.DATA: {},
        }

        assert actual == expected


class TestSerializeVideoLength:
    @pytest.mark.parametrize(
        "given_duration, expected",
        [
            (
                timedelta(hours=2, seconds=1, microseconds=30, milliseconds=50),
                "2:00:01",
            ),
            (
                timedelta(
                    days=1,
                    hours=29,
                    minutes=90,
                    seconds=29,
                    milliseconds=1500,
                ),
                "54:30:30",
            ),
        ],
    )
    def test_format(self, given_duration: timedelta, expected: str) -> None:
        """
        #Bug https://openproject.platomo.de/wp/7647
        """
        actual = serialize_video_length(given_duration)
        assert actual == expected


class TestParseVideoLength:
    @pytest.mark.parametrize(
        "given_duration, expected",
        [
            ("54:30:30", timedelta(hours=54, minutes=30, seconds=30)),
            ("2:00:01.1023", timedelta(hours=2, minutes=0, seconds=1)),
        ],
    )
    def test_parse(self, given_duration: str, expected: timedelta) -> None:
        """
        #Bug https://openproject.platomo.de/wp/7647
        """
        actual = parse_video_length(given_duration)
        assert actual == expected
