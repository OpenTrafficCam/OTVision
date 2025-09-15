from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from OTVision import dataformat, version
from OTVision.application.track.ottrk import (
    OttrkBuilder,
    OttrkBuilderConfig,
    OttrkBuilderError,
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
TRACK_1 = 1
TRACK_2 = 2
FRAME_1 = 1
FRAME_2 = 2
FRAME_1_OCCURRENCE = RECORDED_START_DATE
FRAME_2_OCCURRENCE = FRAME_1_OCCURRENCE + timedelta(seconds=1)
SOURCE = "test_video.mp4"
OUTPUT = "my_output"


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
        target.set_config(given.ottrk_builder_config)

        # Create three tracks with different states
        unfinished_track_id = TrackId(1)
        discarded_track_id = TrackId(2)
        finished_track_id = TrackId(3)

        # Create detections for each track
        first_unfinished_detection = create_tracked_detection(
            unfinished_track_id, is_first=True, is_last=False, is_discarded=False
        )
        first_discarded_detection = create_tracked_detection(
            discarded_track_id, is_first=True, is_last=False, is_discarded=False
        )
        first_finished_detection = create_tracked_detection(
            finished_track_id, is_first=True, is_last=False, is_discarded=False
        )
        second_finished_detection = create_tracked_detection(
            finished_track_id, is_first=False, is_last=True, is_discarded=False
        )

        # Create tracked frame with all detections
        first_frame = create_tracked_frame(
            frame_no=FRAME_1,
            occurrence=RECORDED_START_DATE,
            detections=[
                first_unfinished_detection,
                first_discarded_detection,
                first_finished_detection,
            ],
            discarded_tracks={discarded_track_id},
            finished_tracks=set(),
            unfinished_tracks={
                unfinished_track_id,
                discarded_track_id,
                finished_track_id,
            },
        )

        second_frame = create_tracked_frame(
            frame_no=FRAME_2,
            occurrence=FRAME_2_OCCURRENCE,
            detections=[second_finished_detection],
            discarded_tracks=set(),
            finished_tracks={finished_track_id},
            unfinished_tracks=set(),
        )
        # Add tracked frames to builder
        target.add_tracked_frames([first_frame, second_frame])

        actual = target.build()
        expected_unfinished = create_expected_tracked_detection(
            first_unfinished_detection, FRAME_1, FRAME_1_OCCURRENCE, False
        )
        expected_first_finished = create_expected_tracked_detection(
            first_finished_detection, FRAME_1, FRAME_1_OCCURRENCE, False
        )
        expected_second_finished = create_expected_tracked_detection(
            second_finished_detection, FRAME_2, FRAME_2_OCCURRENCE, True
        )
        expected_track_data_block = [
            expected_unfinished,
            expected_first_finished,
            expected_second_finished,
        ]
        expected_ottrk = create_expected_ottrk(
            actual_duration, expected_duration, expected_track_data_block
        )

        assert actual == expected_ottrk
        assert not target._tracked_detections

    def test_add_config_raises_error_if_config_is_not_set(self) -> None:
        given = setup(create_given(ACTUAL_DURATION, EXPECTED_DURATION))
        target = create_target(given)

        # Don't add config, just try to access it
        with pytest.raises(OttrkBuilderError, match="Ottrk builder config is not set"):
            _ = target.config

    def test_add_config_sets_build_config(self) -> None:
        given = setup(create_given(ACTUAL_DURATION, EXPECTED_DURATION))
        target = create_target(given)

        target.set_config(given.ottrk_builder_config)

        assert target.config == given.ottrk_builder_config

    def test_finish_track_marks_detection_as_finished(self) -> None:
        given = setup(create_given(ACTUAL_DURATION, EXPECTED_DURATION))
        target = create_target(given)
        target.set_config(given.ottrk_builder_config)
        detection_1 = create_tracked_detection(
            1, is_first=True, is_last=False, is_discarded=False
        )
        detection_2 = create_tracked_detection(
            1, is_first=False, is_last=False, is_discarded=False
        )

        frame_1 = create_tracked_frame(
            frame_no=FRAME_1,
            occurrence=RECORDED_START_DATE,
            detections=[detection_1],
            discarded_tracks=set(),
            finished_tracks=set(),
            unfinished_tracks={TRACK_1},
        )
        frame_2 = create_tracked_frame(
            frame_no=FRAME_2,
            occurrence=RECORDED_START_DATE + timedelta(seconds=1),
            detections=[detection_2],
            discarded_tracks=set(),
            finished_tracks=set(),
            unfinished_tracks={TRACK_1},
        )

        target.add_tracked_frames([frame_2, frame_1])
        target.finish_track(TRACK_1)

        result = target.build()
        detections = result[dataformat.DATA][dataformat.DETECTIONS]

        # Find the last detection for this track
        track_detections = [d for d in detections if d[dataformat.TRACK_ID] == TRACK_1]
        assert len(track_detections) == 2

        # The last detection should be marked as finished
        last_detection = max(track_detections, key=lambda d: d[dataformat.OCCURRENCE])
        assert last_detection[dataformat.FINISHED] is True

        # The first detection should not be marked as finished
        first_detection = min(track_detections, key=lambda d: d[dataformat.OCCURRENCE])
        assert first_detection[dataformat.FINISHED] is False

    def test_discard_track_removes(self) -> None:
        given = setup(create_given(ACTUAL_DURATION, EXPECTED_DURATION))
        target = create_target(given)
        target.set_config(given.ottrk_builder_config)

        # Create test data for two tracks
        track_id1 = TrackId(1)  # This one will be discarded
        track_id2 = TrackId(2)  # This one will remain

        detection1 = create_tracked_detection(
            track_id1, is_first=True, is_last=False, is_discarded=False
        )
        detection2 = create_tracked_detection(
            track_id2, is_first=True, is_last=False, is_discarded=False
        )

        frame = create_tracked_frame(
            frame_no=FRAME_1,
            occurrence=RECORDED_START_DATE,
            detections=[detection1, detection2],
            discarded_tracks=set(),
            finished_tracks=set(),
            unfinished_tracks={track_id1, track_id2},
        )

        target.add_tracked_frames([frame])
        target.discard_track(track_id1)

        result = target.build()
        detections = result[dataformat.DATA][dataformat.DETECTIONS]

        # Only track 2 should remain
        track_ids = {d[dataformat.TRACK_ID] for d in detections}
        assert track_ids == {track_id2}
        assert len(detections) == 1


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
        source=SOURCE,
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
            dataformat.TRACKING_RUN_ID: config.tracking_run_id,
            dataformat.FRAME_GROUP: config.frame_group,
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
    result.source = SOURCE

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
    result.source = SOURCE
    result.output = OUTPUT
    result.detections = detections
    result.image = None
    result.discarded_tracks = discarded_tracks
    result.finished_tracks = finished_tracks
    result.unfinished_tracks = unfinished_tracks
    return result


def create_expected_tracked_detection(
    detection: TrackedDetection,
    frame_no: FrameNo,
    occurrence: datetime,
    is_finished: bool,
) -> dict:
    """Create an expected detection dictionary from a TrackedDetection."""
    return {
        dataformat.CLASS: detection.label,
        dataformat.CONFIDENCE: detection.conf,
        dataformat.X: detection.x,
        dataformat.Y: detection.y,
        dataformat.W: detection.w,
        dataformat.H: detection.h,
        dataformat.INTERPOLATED_DETECTION: False,
        dataformat.FIRST: detection.is_first,
        dataformat.FINISHED: is_finished,
        dataformat.TRACK_ID: detection.track_id,
        dataformat.FRAME: frame_no,
        dataformat.OCCURRENCE: occurrence.timestamp(),
        dataformat.INPUT_FILE_PATH: SOURCE,
    }


def create_expected_ottrk(
    actual_duration: timedelta,
    expected_duration: timedelta | None,
    expected_tracked_detections: list[dict],
) -> dict:
    otdet_metadata = create_otdet_metadata(actual_duration, expected_duration)
    track_metadata = create_expected_track_metadata(actual_duration, expected_duration)

    # Add the missing fields that OttrkBuilder adds
    combined_metadata = {
        **otdet_metadata,
        **track_metadata,
    }

    return {
        dataformat.METADATA: combined_metadata,
        dataformat.DATA: {
            dataformat.DETECTIONS: expected_tracked_detections,
        },
    }
