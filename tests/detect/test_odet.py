from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from OTVision import dataformat, version
from OTVision.detect.otdet import OtdetBuilder, OtdetBuilderConfig, OtdetBuilderError
from OTVision.track.preprocess import Detection


def create_expected_video_metadata(config: OtdetBuilderConfig) -> dict:
    """Create the expected video metadata based on the given config."""
    video_metadata = {
        dataformat.FILENAME: str(config.video.stem),
        dataformat.FILETYPE: str(config.video.suffix),
        dataformat.WIDTH: config.video_width,
        dataformat.HEIGHT: config.video_height,
        dataformat.RECORDED_FPS: config.recorded_fps,
        dataformat.ACTUAL_FPS: config.actual_fps,
        dataformat.NUMBER_OF_FRAMES: config.actual_frames,
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


def create_expected_metadata(config: OtdetBuilderConfig) -> dict:
    """Create the full expected metadata based on the given config."""
    return {
        dataformat.OTDET_VERSION: version.otdet_version(),
        dataformat.VIDEO: create_expected_video_metadata(config),
        dataformat.DETECTION: create_expected_detection_metadata(config),
    }


def create_expected_data(
    detections: list[list[Detection]],
) -> dict:
    """Create the expected data dictionary based on input detections."""
    data = {}
    for frame, detection_list in enumerate(detections, start=1):
        data[str(frame)] = {
            dataformat.DETECTIONS: [d.to_otdet() for d in detection_list]
        }
    return data


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
            video=Path("video.mp4"),
            video_width=1920,
            video_height=1080,
            expected_duration=timedelta(seconds=300),
            recorded_fps=30.0,
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

    @pytest.fixture
    def mock_detection(self) -> MagicMock:
        """Fixture to provide a mocked Detection object."""
        detection: MagicMock = MagicMock(spec=Detection)
        detection.to_otdet.return_value = {
            dataformat.CLASS: "person",
            dataformat.CONFIDENCE: 0.9,
        }
        return detection

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
        actual = builder._build_metadata()

        expected = create_expected_metadata(config)
        assert actual == expected

    def test_build_data(
        self,
        builder: OtdetBuilder,
        config: OtdetBuilderConfig,
        mock_detection: MagicMock,
    ) -> None:
        """Test the data generation with detection objects.

        https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7188

        """  # noqa
        builder.add_config(config)

        # Provide mock detections as input
        given: list[list[Detection]] = [[mock_detection] * 2, [mock_detection]]
        actual = builder._build_data(given)

        expected = create_expected_data(given)
        assert actual == expected

    def test_build_full_result(
        self,
        builder: OtdetBuilder,
        config: OtdetBuilderConfig,
        mock_detection: MagicMock,
    ) -> None:
        """Test the full build method.


        https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7188
        """  # noqa
        builder.add_config(config)

        # Provide mock detections
        given: list[list[Detection]] = [[mock_detection] * 3]
        actual = builder.build(given)

        expected_metadata = create_expected_metadata(config)
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
            dataformat.METADATA: create_expected_metadata(config),
            dataformat.DATA: {},
        }

        assert actual == expected
