from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from OTVision.application.buffer import Buffer
from OTVision.application.config import Config, TrackConfig
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.application.otvision_save_path_provider import OtvisionSavePathProvider
from OTVision.application.track.ottrk import OttrkBuilder, OttrkBuilderConfig
from OTVision.application.track.tracking_run_id import GetCurrentTrackingRunId
from OTVision.detect.otdet import OtdetBuilderConfig
from OTVision.detect.otdet_file_writer import OtdetFileWrittenEvent
from OTVision.domain.detection import TrackId
from OTVision.domain.frame import TrackedFrame
from OTVision.track.stream_ottrk_file_writer import (
    STREAMING_FRAME_GROUP_ID,
    OttrkFileWrittenEvent,
    StreamOttrkFileWriter,
)

# Test data constants
TEST_SOURCE = "test_video.mp4"
TEST_TRACKING_RUN_ID = "test_run_001"
TEST_OUTPUT_PATH = Path("/test/output/tracks.ottrk")
RECORDED_START_DATE = datetime(2023, 1, 1, 12, 0, 0)
ACTUAL_DURATION = timedelta(seconds=295)
EXPECTED_DURATION = timedelta(seconds=300)
TRACK_ID_1 = TrackId(1)
TRACK_ID_2 = TrackId(2)
TRACK_ID_3 = TrackId(3)
TRACK_ID_4 = TrackId(4)


class TestStreamOttrkFileWriter:
    """Test suite for StreamOttrkFileWriter following stage play principles.

    Each test is structured as a stage play with:
    - Given (Arrange): Set up the test scenario
    - When (Act): Execute the action being tested
    - Then (Assert): Verify the expected outcome
    """

    def test_inherits_from_buffer(self) -> None:
        # Given: A StreamOttrkFileWriter instance
        given = create_given()
        target = create_target(given)

        # Then: It should be an instance of Buffer
        assert isinstance(target, Buffer)

    def test_config_property_returns_current_config(self) -> None:
        # Given: A configured StreamOttrkFileWriter
        given = create_given()
        target = create_target(given)

        # When: Accessing the config property
        actual_config = target.config

        # Then: It should return the config from get_current_config
        assert actual_config == given.config
        given.get_current_config.get.assert_called_once()

    def test_track_config_property_returns_track_config(self) -> None:
        # Given: A configured StreamOttrkFileWriter
        given = create_given()
        target = create_target(given)

        # When: Accessing the track_config property
        actual_track_config = target.track_config

        # Then: It should return the track config from the main config
        assert actual_track_config == given.config.track

    def test_build_condition_fulfilled_when_no_unfinished_tracks(self) -> None:
        # Given: A StreamOttrkFileWriter with no unfinished tracks
        given = create_given()
        target = create_target(given)

        # When: Checking if build condition is fulfilled
        is_fulfilled = target.build_condition_fulfilled

        # Then: It should return True (no unfinished tracks initially)
        assert is_fulfilled is True

    def test_build_condition_not_fulfilled_when_unfinished_tracks_exist(self) -> None:
        # Given: A StreamOttrkFileWriter with unfinished tracks
        given = create_given()
        target = create_target(given)

        # When: Adding unfinished tracks and checking build condition
        target._ottrk_unfinished_tracks.add(TrackId(1))
        is_fulfilled = target.build_condition_fulfilled

        # Then: It should return False
        assert is_fulfilled is False

    def test_current_output_file_raises_error_when_not_set(self) -> None:
        # Given: A StreamOttrkFileWriter without output file set
        given = create_given()
        target = create_target(given)

        # When/Then: Accessing current_output_file should raise ValueError
        with pytest.raises(ValueError, match="Output file has not been set yet"):
            _ = target.current_output_file

    def test_current_output_file_returns_path_when_set(self) -> None:
        # Given: A StreamOttrkFileWriter with output file set
        given = create_given()
        target = create_target(given)
        target._current_output_file = TEST_OUTPUT_PATH

        # When: Accessing current_output_file
        actual_path = target.current_output_file

        # Then: It should return the set path
        assert actual_path == TEST_OUTPUT_PATH

    def test_buffer_adds_frame_without_image(self) -> None:
        # Given: A StreamOttrkFileWriter and a tracked frame
        given = create_given()
        target = create_target(given)
        tracked_frame = create_tracked_frame()

        # When: Buffering the tracked frame
        target.buffer(tracked_frame)

        # Then: The frame should be added to buffer without image
        buffered_elements = target._get_buffered_elements()
        assert len(buffered_elements) == 1
        tracked_frame.without_image.assert_called_once()

    @patch("OTVision.track.stream_ottrk_file_writer.write_json")
    def test_buffer_processes_tracks_when_in_writing_state(
        self, mock_write_json: Mock
    ) -> None:
        # Given: A StreamOttrkFileWriter in writing state with unfinished tracks
        given = create_given()
        target = create_target(given)
        target._in_writing_state = True
        target._current_output_file = TEST_OUTPUT_PATH
        target._ottrk_unfinished_tracks.update(
            {TRACK_ID_1, TRACK_ID_2, TRACK_ID_3, TRACK_ID_4}
        )  # Add unfinished track to prevent build
        tracked_frame = create_tracked_frame(
            finished_tracks={TRACK_ID_1},
            discarded_tracks={TRACK_ID_2},
            unfinished_tracks={TRACK_ID_3},
        )

        # When: Buffering a tracked frame
        target.buffer(tracked_frame)

        # Then: It should process finished and discarded tracks
        given.builder.finish_tracks.assert_called_once_with(
            tracked_frame.finished_tracks
        )
        given.builder.discard_tracks.assert_called_once_with(
            tracked_frame.discarded_tracks
        )
        given.builder.build.assert_not_called()
        assert target._ottrk_unfinished_tracks == {TRACK_ID_4}
        mock_write_json.assert_not_called()

    @patch("OTVision.track.stream_ottrk_file_writer.write_json")
    def test_buffer_builds_and_writes_when_condition_fulfilled(
        self, mock_write_json: Mock
    ) -> None:
        # Given: A StreamOttrkFileWriter in writing state with unfinished tracks
        given = create_given()
        target = create_target(given)
        target._in_writing_state = True
        target._current_output_file = TEST_OUTPUT_PATH
        target._ottrk_unfinished_tracks.add(
            TRACK_ID_1
        )  # Add unfinished track to prevent build
        tracked_frame = create_tracked_frame(
            finished_tracks={TRACK_ID_1}, discarded_tracks={TRACK_ID_2}
        )

        # When: Buffering a tracked frame
        target.buffer(tracked_frame)

        # Then: It should process finished and discarded tracks
        given.builder.finish_tracks.assert_called_once_with(
            tracked_frame.finished_tracks
        )
        given.builder.discard_tracks.assert_called_once_with(
            tracked_frame.discarded_tracks
        )
        given.builder.build.assert_called_once()
        assert len(target._ottrk_unfinished_tracks) == 0
        mock_write_json.assert_called_once_with(
            dict_to_write=given.built_ottrk,
            file=TEST_OUTPUT_PATH,
            filetype=given.config.filetypes.track,
            overwrite=True,
        )
        assert target._in_writing_state is False

    def test_on_flush_returns_early_when_no_tracked_frames(self) -> None:
        # Given: A StreamOttrkFileWriter with no buffered frames
        given = create_given()
        target = create_target(given)
        event = create_otdet_file_written_event()

        # When: Calling on_flush
        target.on_flush(event)

        # Then: It should return early without processing
        given.save_path_provider.provide.assert_not_called()
        given.builder.set_config.assert_not_called()
        given.builder.add_tracked_frames.assert_not_called()
        given.builder.build.assert_not_called()
        assert target._in_writing_state is False

    def test_on_flush_sets_writing_state_and_configures_builder(self) -> None:
        # Given: A StreamOttrkFileWriter with buffered frames
        given = create_given()
        target = create_target(given)

        first_tracked_frame = create_tracked_frame()
        last_tracked_frame = create_tracked_frame(
            unfinished_tracks={TRACK_ID_2},
            discarded_tracks={TRACK_ID_3},
            finished_tracks={TRACK_ID_4},
        )
        target.buffer(first_tracked_frame)
        target.buffer(last_tracked_frame)
        event = create_otdet_file_written_event()

        # When: Calling on_flush
        target.on_flush(event)

        # Then: It should set up writing state and configure builder
        assert target._in_writing_state is True
        given.save_path_provider.provide.assert_called_once_with(
            event.otdet_builder_config.source, given.config.filetypes.track
        )
        given.builder.set_config.assert_called_once_with(
            create_expected_builder_config(
                track_config=given.config.track,
                otdet_builder_config=event.otdet_builder_config,
                number_of_frames=event.number_of_frames,
                tracking_run_id=TEST_TRACKING_RUN_ID,
            )
        )
        given.builder.add_tracked_frames.assert_called_once_with(
            [first_tracked_frame, last_tracked_frame]
        )
        assert target._ottrk_unfinished_tracks == {TRACK_ID_2}

    def test_on_flush_creates_correct_ottrk_builder_config(self) -> None:
        # Given: A StreamOttrkFileWriter with buffered frames
        given = create_given()
        target = create_target(given)
        tracked_frame = create_tracked_frame()
        target.buffer(tracked_frame)
        event = create_otdet_file_written_event()

        # When: Calling on_flush
        target.on_flush(event)

        # Then: It should create correct builder config
        call_args = given.builder.set_config.call_args[0][0]
        assert isinstance(call_args, OttrkBuilderConfig)
        assert call_args.otdet_builder_config == event.otdet_builder_config
        assert call_args.number_of_frames == event.number_of_frames
        assert call_args.sigma_l == given.config.track.sigma_l
        assert call_args.sigma_h == given.config.track.sigma_h
        assert call_args.sigma_iou == given.config.track.sigma_iou
        assert call_args.t_min == given.config.track.t_min
        assert call_args.t_miss_max == given.config.track.t_miss_max
        assert call_args.tracking_run_id == TEST_TRACKING_RUN_ID
        assert call_args.frame_group == STREAMING_FRAME_GROUP_ID

    def test_reset_clears_buffer(self) -> None:
        """Verify that reset clears the buffer."""
        # Given: A StreamOttrkFileWriter with buffered data
        given = create_given()
        target = create_target(given)
        tracked_frame = create_tracked_frame()
        target.buffer(tracked_frame)

        # When: Calling reset
        target.reset()

        # Then: Buffer should be cleared
        assert len(target._get_buffered_elements()) == 0

    @patch("OTVision.track.stream_ottrk_file_writer.write_json")
    def test_write_calls_write_json_with_correct_parameters(
        self, mock_write_json: Mock
    ) -> None:
        """Verify that write method calls write_json with correct parameters."""
        # Given: A StreamOttrkFileWriter with output file set
        given = create_given()
        target = create_target(given)
        target._current_output_file = TEST_OUTPUT_PATH
        ottrk_data = {"test": "data"}

        # When: Calling write
        target.write(ottrk_data)

        # Then: It should call write_json with correct parameters
        mock_write_json.assert_called_once_with(
            dict_to_write=ottrk_data,
            file=TEST_OUTPUT_PATH,
            filetype=given.config.filetypes.track,
            overwrite=True,
        )
        given.subject.notify.assert_called_once_with(
            OttrkFileWrittenEvent(save_location=TEST_OUTPUT_PATH)
        )


@dataclass
class Given:
    """Test data container following the Given-When-Then pattern."""

    builder: Mock
    built_ottrk: Mock
    get_current_config: Mock
    get_current_tracking_run_id: Mock
    save_path_provider: Mock
    config: Mock
    subject: Mock


def create_given() -> Given:
    """Create test data following the Given-When-Then pattern."""
    # Mock dependencies
    built_ottrk = Mock()
    builder = Mock(spec=OttrkBuilder)
    builder.build.return_value = built_ottrk

    get_current_config = Mock(spec=GetCurrentConfig)
    get_current_tracking_run_id = Mock(spec=GetCurrentTrackingRunId)
    save_path_provider = Mock(spec=OtvisionSavePathProvider)

    # Mock configuration
    config = Mock(spec=Config)
    track_config = Mock(spec=TrackConfig)
    track_config.sigma_l = 0.3
    track_config.sigma_h = 0.7
    track_config.sigma_iou = 0.5
    track_config.t_min = 5
    track_config.t_miss_max = 10
    config.track = track_config
    config.filetypes.track = "ottrk"

    # Configure mocks
    get_current_config.get.return_value = config
    get_current_tracking_run_id.get.return_value = TEST_TRACKING_RUN_ID
    save_path_provider.provide.return_value = TEST_OUTPUT_PATH

    return Given(
        builder=builder,
        built_ottrk=built_ottrk,
        get_current_config=get_current_config,
        get_current_tracking_run_id=get_current_tracking_run_id,
        save_path_provider=save_path_provider,
        config=config,
        subject=Mock(),
    )


def create_target(given: Given) -> StreamOttrkFileWriter:
    """Create the target StreamOttrkFileWriter instance."""
    return StreamOttrkFileWriter(
        builder=given.builder,
        get_current_config=given.get_current_config,
        get_current_tracking_run_id=given.get_current_tracking_run_id,
        save_path_provider=given.save_path_provider,
        subject=given.subject,
    )


def create_tracked_frame(
    finished_tracks: set[TrackId] | None = None,
    discarded_tracks: set[TrackId] | None = None,
    unfinished_tracks: set[TrackId] | None = None,
) -> Mock:
    """Create a mock TrackedFrame for testing."""
    frame = Mock(spec=TrackedFrame)
    frame.finished_tracks = finished_tracks or set()
    frame.discarded_tracks = discarded_tracks or set()
    frame.unfinished_tracks = unfinished_tracks or set()
    frame.without_image.return_value = frame
    return frame


def create_otdet_file_written_event() -> Mock:
    """Create a mock OtdetFileWrittenEvent for testing."""
    event = Mock(spec=OtdetFileWrittenEvent)
    event.otdet_builder_config = Mock(spec=OtdetBuilderConfig)
    event.otdet_builder_config.source = TEST_SOURCE
    event.number_of_frames = 100
    return event


def create_expected_builder_config(
    track_config: TrackConfig,
    otdet_builder_config: OtdetBuilderConfig,
    number_of_frames: int,
    tracking_run_id: str,
) -> OttrkBuilderConfig:
    return OttrkBuilderConfig(
        otdet_builder_config=otdet_builder_config,
        number_of_frames=number_of_frames,
        sigma_l=track_config.sigma_l,
        sigma_h=track_config.sigma_h,
        sigma_iou=track_config.sigma_iou,
        t_min=track_config.t_min,
        t_miss_max=track_config.t_miss_max,
        tracking_run_id=tracking_run_id,
        frame_group=STREAMING_FRAME_GROUP_ID,
    )
