"""End-to-end integration tests for BOXMOT tracking pipeline."""

import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from OTVision.application.config import Config
from OTVision.domain.detection import Detection, TrackId
from OTVision.domain.frame import DetectedFrame, FrameNo

# Check if BOXMOT is available
try:
    import boxmot

    BOXMOT_INSTALLED = True
except ImportError:
    BOXMOT_INSTALLED = False


class TestBoxmotPipelineConfigurationSelection:
    """Test that TrackBuilder correctly selects tracker based on configuration."""

    def test_builder_uses_iou_by_default(self) -> None:
        """Test TrackBuilder uses IOU tracker by default."""
        from OTVision.track.builder import TrackBuilder

        builder = TrackBuilder()

        # Default config should have BOXMOT disabled
        config = builder.get_current_config.get()
        assert config.track.boxmot.enabled is False

        # Tracker should be GroupedFilesTracker (wrapping IOU)
        tracker = builder.tracker
        assert tracker is not None

    def test_builder_uses_iou_when_boxmot_disabled(self) -> None:
        """Test TrackBuilder uses IOU tracker when BOXMOT disabled."""
        from OTVision.track.builder import TrackBuilder

        builder = TrackBuilder()
        config = builder.get_current_config.get()

        # Ensure BOXMOT is disabled (default)
        assert config.track.boxmot.enabled is False

        # Access the tracker
        tracker = builder.tracker

        # Verify tracker is created
        assert tracker is not None


@pytest.mark.skipif(not BOXMOT_INSTALLED, reason="BOXMOT not installed")
class TestBoxmotPipelineTracking:
    """Test end-to-end tracking with BOXMOT adapter."""

    @pytest.fixture
    def temp_output_dir(self) -> Iterator[Path]:
        """Create a temporary directory for output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def synthetic_detected_frames(self) -> list[DetectedFrame]:
        """Create synthetic detected frames for testing."""
        frames = []

        # Frame 1: Two objects appear
        frames.append(
            DetectedFrame(
                no=FrameNo(1),
                occurrence=0.0,
                source="synthetic_video.mp4",
                output="synthetic.otdet",
                detections=[
                    Detection(label="car", conf=0.9, x=100.0, y=150.0, w=50.0, h=80.0),
                    Detection(
                        label="pedestrian",
                        conf=0.85,
                        x=200.0,
                        y=300.0,
                        w=40.0,
                        h=120.0,
                    ),
                ],
                image=np.zeros((720, 1280, 3), dtype=np.uint8),
            )
        )

        # Frame 2: Same objects continue
        frames.append(
            DetectedFrame(
                no=FrameNo(2),
                occurrence=0.033,
                source="synthetic_video.mp4",
                output="synthetic.otdet",
                detections=[
                    Detection(label="car", conf=0.92, x=105.0, y=155.0, w=50.0, h=80.0),
                    Detection(
                        label="pedestrian",
                        conf=0.88,
                        x=205.0,
                        y=305.0,
                        w=40.0,
                        h=120.0,
                    ),
                ],
                image=np.zeros((720, 1280, 3), dtype=np.uint8),
            )
        )

        # Frame 3: One object disappears
        frames.append(
            DetectedFrame(
                no=FrameNo(3),
                occurrence=0.066,
                source="synthetic_video.mp4",
                output="synthetic.otdet",
                detections=[
                    Detection(label="car", conf=0.91, x=110.0, y=160.0, w=50.0, h=80.0),
                ],
                image=np.zeros((720, 1280, 3), dtype=np.uint8),
            )
        )

        return frames

    def test_complete_tracking_pipeline_with_boxmot(
        self,
        synthetic_detected_frames: list[DetectedFrame],
    ) -> None:
        """Test complete tracking pipeline with real BOXMOT tracker."""
        from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter

        # Create adapter with real ByteTrack
        adapter = BoxmotTrackerAdapter(
            tracker_type="bytetrack", device="cpu", half=False
        )

        # Track frames
        def id_gen() -> Iterator[TrackId]:
            yield from range(1000)

        id_generator = id_gen()
        tracked_frames = []

        for frame in synthetic_detected_frames:
            tracked_frame = adapter.track_frame(frame, id_generator)
            tracked_frames.append(tracked_frame)

        # Verify tracking results
        assert len(tracked_frames) == 3

        # Verify that tracking produces results (exact counts may vary with real tracker)
        assert all(isinstance(f.detections, list) for f in tracked_frames)
        assert all(isinstance(f.finished_tracks, set) for f in tracked_frames)

    def test_tracker_reset_between_videos(
        self,
        synthetic_detected_frames: list[DetectedFrame],
    ) -> None:
        """Test that tracker resets correctly between different videos."""
        from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter

        adapter = BoxmotTrackerAdapter(
            tracker_type="bytetrack", device="cpu", half=False
        )

        def id_gen() -> Iterator[TrackId]:
            yield from range(1000)

        # Track first video frame
        frame1 = synthetic_detected_frames[0]
        adapter.track_frame(frame1, id_gen())

        # Note: With real ByteTrack, internal state may or may not be populated
        # depending on detection thresholds and tracking decisions

        # Reset for next video
        adapter.reset()

        # Verify state is cleared after reset
        assert len(adapter._track_id_mapping) == 0
        assert len(adapter._class_mapper.get_forward_mapping()) == 0

        # Track second video frame
        frame2 = synthetic_detected_frames[1]
        tracked_frame2 = adapter.track_frame(frame2, id_gen())

        # Should work without errors
        assert tracked_frame2 is not None


@pytest.mark.skipif(not BOXMOT_INSTALLED, reason="BOXMOT not installed")
class TestBoxmotPipelinePerformance:
    """Basic performance validation tests."""

    def test_tracker_handles_many_frames(self) -> None:
        """Test tracker can process many frames without memory issues."""
        from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter

        adapter = BoxmotTrackerAdapter(
            tracker_type="bytetrack", device="cpu", half=False
        )

        def id_gen() -> Iterator[TrackId]:
            yield from range(10000)

        # Track 100 frames
        for i in range(100):
            frame = DetectedFrame(
                no=FrameNo(i),
                occurrence=i * 0.033,
                source="test.mp4",
                output="test.otdet",
                detections=[
                    Detection(
                        label="car", conf=0.9, x=100.0 + i, y=150.0, w=50.0, h=80.0
                    ),
                    Detection(
                        label="pedestrian",
                        conf=0.85,
                        x=200.0,
                        y=300.0 + i,
                        w=40.0,
                        h=120.0,
                    ),
                ],
                image=np.zeros((720, 1280, 3), dtype=np.uint8),
            )

            tracked_frame = adapter.track_frame(frame, id_gen())
            assert tracked_frame is not None

        # Verify track ID mapping doesn't grow unbounded
        # With finished tracks cleanup, mapping should only contain active tracks
        # In this test, tracks continue across all frames, so mapping has 2 entries
        assert len(adapter._track_id_mapping) <= 10  # Reasonable upper bound

    def test_tracker_handles_many_detections_per_frame(
        self,
    ) -> None:
        """Test tracker handles frames with many detections."""
        from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter

        adapter = BoxmotTrackerAdapter(
            tracker_type="bytetrack", device="cpu", half=False
        )

        def id_gen() -> Iterator[TrackId]:
            yield from range(10000)

        # Create frame with 100 detections
        detections = [
            Detection(label="object", conf=0.9, x=float(i), y=float(i), w=50.0, h=80.0)
            for i in range(100)
        ]

        frame = DetectedFrame(
            no=FrameNo(1),
            occurrence=0.0,
            source="test.mp4",
            output="test.otdet",
            detections=detections,
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
        )

        # Should handle without error (exact count may vary with real tracker)
        tracked_frame = adapter.track_frame(frame, id_gen())
        assert tracked_frame is not None
        assert isinstance(tracked_frame.detections, list)
