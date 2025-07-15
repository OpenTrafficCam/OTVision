"""Unit tests for OttrkBuilder following stage play principles.

This module contains comprehensive tests for the OttrkBuilder class,
structured as stage plays with clear Act/Arrange/Assert patterns.
Each test tells a story about the behavior being tested.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from OTVision import dataformat, version
from OTVision.application.track.ottrk import (
    OttrkBuilder,
    OttrkBuilderConfig,
    create_tracker_metadata,
)
from OTVision.detect.otdet import (
    OtdetBuilderConfig,
    OtdetMetadataBuilder,
    serialize_video_length,
)
from OTVision.domain.detection import TrackedDetection, TrackId
from OTVision.domain.frame import FrameNo, TrackedFrame

RECORDED_START_DATE = datetime(2023, 1, 1, 12, 0, 0)
ACTUAL_DURATION = timedelta(seconds=295)
EXPECTED_DURATION = timedelta(seconds=300)


class TestOttrkBuilder:
    @pytest.mark.parametrize(
        "actual_duration,expected_duration,expected_start_date,expected_end_date",
        [
            (
                ACTUAL_DURATION,
                EXPECTED_DURATION,
                RECORDED_START_DATE,
                RECORDED_START_DATE + EXPECTED_DURATION,
            ),
            (
                ACTUAL_DURATION,
                None,
                RECORDED_START_DATE,
                RECORDED_START_DATE + ACTUAL_DURATION,
            ),
        ],
    )
    def test_build_correctly_configured_config(
        self,
        actual_duration: timedelta,
        expected_duration: timedelta | None,
        expected_start_date: datetime,
        expected_end_date: datetime,
    ) -> None:
        given = setup(create_given(actual_duration, expected_duration))

        target = create_target(given)
        target.add_config(given.ottrk_builder_config)

        actual = target.build()  # noqa

    def test_add_config_raises_error_if_config_is_not_set(self) -> None:
        raise NotImplementedError

    def test_add_config_sets_build_config(self) -> None:
        raise NotImplementedError

    def test_finish_track_marks_detection_as_finished(self) -> None:
        raise NotImplementedError

    def test_finish_tracks_marks_detections_as_finished(self) -> None:
        raise NotImplementedError

    def test_discard_tracks_removes(self) -> None:
        raise NotImplementedError

    def test_discard_track_removes(self) -> None:
        raise NotImplementedError


@dataclass
class Given:
    otdet_metadata_builder: Mock
    otdet_metadata: dict
    actual_duration: timedelta
    expected_duration: timedelta | None
    ottrk_builder_config: OttrkBuilderConfig


def setup(given: Given) -> Given:
    given.otdet_metadata_builder.build.return_value = given.otdet_metadata
    return given


def create_given(
    actual_duration: timedelta, expected_duration: timedelta | None
) -> Given:

    return Given(
        otdet_metadata_builder=Mock(spec=OtdetMetadataBuilder),
        otdet_metadata=create_otdet_metadata(actual_duration, expected_duration),
        actual_duration=actual_duration,
        expected_duration=expected_duration,
        ottrk_builder_config=create_ottrk_builder_config(
            actual_duration, expected_duration
        ),
    )


def create_target(given: Given) -> OttrkBuilder:
    return OttrkBuilder(given.otdet_metadata_builder)


def create_ottrk_builder_config(
    actual_duration: timedelta, expected_duration: timedelta | None
) -> OttrkBuilderConfig:
    otdet_builder_config = create_otdet_builder_config(
        actual_duration, expected_duration
    )
    return OttrkBuilderConfig(
        otdet_builder_config=otdet_builder_config,
        number_of_frames=100,
        sigma_l=0.3,
        sigma_h=0.7,
        sigma_iou=0.5,
        t_min=5,
        t_miss_max=10,
        tracking_run_id="test_run_001",
        frame_group=1,
    )


def create_otdet_builder_config(
    actual_duration: timedelta, expected_duration: timedelta | None
) -> OtdetBuilderConfig:
    return OtdetBuilderConfig(
        conf=0.5,
        iou=0.4,
        source="test_video.mp4",
        video_width=1920,
        video_height=1080,
        expected_duration=expected_duration,
        actual_duration=actual_duration,
        recorded_fps=30.0,
        recorded_start_date=RECORDED_START_DATE,
        actual_fps=29.97,
        actual_frames=8850,
        detection_img_size=640,
        normalized=True,
        detection_model=Path("yolov8.pt"),
        half_precision=False,
        chunksize=32,
        classifications={0: "person", 1: "bicycle", 2: "car"},
        detect_start=0,
        detect_end=8850,
    )


def create_otdet_metadata(
    actual_duration: timedelta, expected_duration: timedelta | None
) -> dict:
    return {
        dataformat.OTDET_VERSION: version.otdet_version(),
        dataformat.VIDEO: {
            dataformat.RECORDED_START_DATE: RECORDED_START_DATE.timestamp(),
            dataformat.EXPECTED_DURATION: (
                expected_duration.total_seconds()
                if expected_duration is not None
                else None
            ),
            dataformat.LENGTH: serialize_video_length(actual_duration),
        },
    }


def create_expected_track_metadata(
    actual_duration: timedelta, expected_duration: timedelta | None
) -> dict:
    config = create_ottrk_builder_config(actual_duration, expected_duration)
    start_date = RECORDED_START_DATE
    duration = actual_duration if expected_duration is None else expected_duration
    end_date = start_date + duration

    return {
        dataformat.OTTRACK_VERSION: version.ottrack_version(),
        dataformat.TRACKING: {
            dataformat.OTVISION_VERSION: version.otvision_version(),
            dataformat.FIRST_TRACKED_VIDEO_START: start_date.timestamp(),
            dataformat.LAST_TRACKED_VIDEO_END: end_date.timestamp(),
            dataformat.TRACKER: create_tracker_metadata(
                config.sigma_l,
                config.sigma_h,
                config.sigma_iou,
                config.t_min,
                config.t_miss_max,
            ),
        },
    }


def create_tracked_detection(
    track_id: TrackId, is_first: bool, is_last: bool, is_discarded: bool
) -> TrackedDetection:
    result = Mock()
    result.track_id = track_id
    result.label = "car"
    result.conf = 0.75
    result.x = 0
    result.y = 1
    result.w = 15
    result.h = 25
    result.is_first = is_first
    result.is_last = is_last
    result.is_discarded = is_discarded

    result.to_otdet.return_value = {
        dataformat.CLASS: result.label,
        dataformat.CONFIDENCE: result.conf,
        dataformat.X: result.x,
        dataformat.Y: result.y,
        dataformat.W: result.w,
        dataformat.H: result.h,
    }
    return result


def create_tracked_frame(
    frame_no: FrameNo,
    occurrence: datetime,
    detections: list[TrackedDetection],
    discarded_tracks: set[TrackId],
    finished_tracks: set[TrackId],
    unfinished_tracks: set[TrackId],
) -> TrackedFrame:
    result = Mock()
    result.no = frame_no
    result.occurrence = occurrence
    result.source = "my_source"
    result.output = "my_output"
    result.detections = detections
    result.image = None
    result.discarded_tracks = discarded_tracks
    result.finished_tracks = finished_tracks
    result.unfinished_tracks = unfinished_tracks
    return result


def create_expected_ottrk(
    actual_duration: timedelta,
    expected_duration: timedelta | None,
) -> dict:
    otdet_metadata = create_otdet_metadata(actual_duration, expected_duration)
    track_metadata = create_expected_track_metadata(actual_duration, expected_duration)
    return {
        dataformat.METADATA: {**otdet_metadata, **track_metadata},
        dataformat.DATA: [],
    }
