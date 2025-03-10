from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from OTVision.application.detect.current_object_detector_metadata import (
    CurrentObjectDetectorMetadata,
)
from OTVision.application.detect.detection_file_save_path_provider import (
    DetectionFileSavePathProvider,
)
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.config import Config, DetectConfig
from OTVision.detect.detected_frame_buffer import (
    DetectedFrameBufferEvent,
    SourceMetadata,
)
from OTVision.detect.otdet import OtdetBuilder, OtdetBuilderConfig
from OTVision.detect.otdet_file_writer import OtdetFileWriter
from OTVision.domain.object_detection import ObjectDetectorMetadata

CLASS_MAPPING = {0: "person", 1: "car"}
SAVE_PATH = Path("output/detections.json")
OTDET = Mock()
EXPECTED_DURATION = timedelta(seconds=10)


class TestOtdetFileWriter:

    @pytest.fixture
    def given_event(self) -> DetectedFrameBufferEvent:
        return DetectedFrameBufferEvent(
            frames=[],
            source_metadata=SourceMetadata(
                source="test_video.mp4",
                width=1920,
                height=1080,
                duration=timedelta(seconds=10),
                fps=30.0,
                start_time=datetime(2024, 1, 1, 12, 0, 0),
            ),
        )

    @pytest.mark.parametrize("expected_duration", [EXPECTED_DURATION, None])
    @patch("OTVision.detect.otdet_file_writer.write_json")
    def test_write_with_expected_duration(
        self,
        mock_write_json: Mock,
        expected_duration: timedelta | None,
        given_event: DetectedFrameBufferEvent,
    ) -> None:
        config = create_config(expected_duration=EXPECTED_DURATION)
        given_otdet_builder = create_otdet_builder()
        given_get_current_config = create_get_current_config(config)
        given_object_detector_metadata = create_object_detector_metadata()
        given_get_object_detector_metadata = create_get_object_detector_metadata(
            given_object_detector_metadata
        )
        given_save_path_provider = create_save_path_provider()

        target = OtdetFileWriter(
            builder=given_otdet_builder,
            get_current_config=given_get_current_config,
            current_object_detector_metadata=given_get_object_detector_metadata,
            save_path_provider=given_save_path_provider,
        )

        target.write(given_event)

        expected_detect_config = config.detect
        expected_source_metadata = given_event.source_metadata
        expected_actual_frames = len(given_event.frames)
        actual_fps = (
            expected_actual_frames / expected_source_metadata.duration.total_seconds()
        )
        if expected_duration is not None:
            actual_fps = expected_actual_frames / expected_duration.total_seconds()
        given_otdet_builder.add_config.assert_called_once_with(
            OtdetBuilderConfig(
                conf=expected_detect_config.confidence,
                iou=expected_detect_config.iou,
                source=expected_source_metadata.source,
                video_width=expected_source_metadata.width,
                video_height=expected_source_metadata.height,
                expected_duration=EXPECTED_DURATION,
                actual_duration=expected_source_metadata.duration,
                recorded_fps=expected_source_metadata.fps,
                recorded_start_date=expected_source_metadata.start_time,
                actual_fps=actual_fps,
                actual_frames=expected_actual_frames,
                detection_img_size=expected_detect_config.img_size,
                normalized=expected_detect_config.normalized,
                detection_model=expected_detect_config.weights,
                half_precision=expected_detect_config.half_precision,
                chunksize=1,
                classifications=CLASS_MAPPING,
                detect_start=expected_detect_config.detect_start,
                detect_end=expected_detect_config.detect_end,
            )
        )
        given_otdet_builder.build.assert_called_once_with(given_event.frames)
        given_save_path_provider.provide.assert_called_once_with(
            expected_source_metadata.source
        )
        mock_write_json.assert_called_once_with(
            OTDET,
            file=SAVE_PATH,
            filetype=config.filetypes.detect,
            overwrite=expected_detect_config.overwrite,
        )


def create_otdet_builder() -> Mock:
    builder = Mock(spec=OtdetBuilder)
    builder.add_config.return_value = builder
    builder.build.return_value = OTDET
    return builder


def create_config(expected_duration: timedelta | None) -> Config:
    return Config(detect=DetectConfig(expected_duration=expected_duration))


def create_get_current_config(config: Config) -> Mock:
    mock = Mock(spec=GetCurrentConfig)
    mock.get.return_value = config
    return mock


def create_object_detector_metadata() -> Mock:
    mock = Mock(spec=ObjectDetectorMetadata)
    type(mock).classifications = CLASS_MAPPING
    return mock


def create_get_object_detector_metadata(object_detector_metadata: Mock) -> Mock:
    mock = Mock(spec=CurrentObjectDetectorMetadata)
    mock.get.return_value = object_detector_metadata
    return mock


def create_save_path_provider() -> Mock:
    mock = Mock(spec=DetectionFileSavePathProvider)
    mock.provide.return_value = SAVE_PATH
    return mock
