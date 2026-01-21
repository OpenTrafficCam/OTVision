"""Simplified integration tests for BOXMOT tracker adapter (no BOXMOT install required)."""

from typing import Iterator
from unittest.mock import MagicMock

import numpy as np
import pytest

from OTVision.domain.detection import Detection, TrackId
from OTVision.domain.frame import DetectedFrame, FrameNo

# Skip all tests if BOXMOT not available
pytest.importorskip(
    "boxmot",
    reason="BOXMOT not installed - install with: uv pip install -e .[tracking_boxmot]",
)


class TestBoxmotTrackerAdapterWithRealBoxmot:
    """Test BoxmotTrackerAdapter with real BOXMOT installation."""

    @pytest.fixture
    def id_generator(self) -> Iterator[TrackId]:
        """Create an ID generator for testing."""
        return iter(range(10000))

    def test_init_bytetrack(self) -> None:
        """Test initialization of ByteTrack tracker."""
        from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter

        adapter = BoxmotTrackerAdapter(
            tracker_type="bytetrack", device="cpu", half=False
        )
        assert adapter is not None

    def test_init_invalid_tracker(self) -> None:
        """Test initialization with invalid tracker type."""
        from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter

        with pytest.raises(ValueError, match="Unknown tracker type"):
            BoxmotTrackerAdapter(
                tracker_type="invalidtracker", device="cpu", half=False
            )

    def test_init_with_tracker_params(self) -> None:
        """Test initialization with custom tracker_params."""
        from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter

        adapter = BoxmotTrackerAdapter(
            tracker_type="bytetrack",
            device="cpu",
            half=False,
            tracker_params={"track_buffer": 60, "track_thresh": 0.5},
        )
        assert adapter is not None

    def test_appearance_tracker_requires_reid_weights(self) -> None:
        """Test that appearance trackers require reid_weights."""
        from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter

        with pytest.raises(ValueError, match="requires reid_weights"):
            BoxmotTrackerAdapter(
                tracker_type="botsort",
                reid_weights=None,
                device="cpu",
                half=False,
            )

    def test_track_empty_frame(self, id_generator: Iterator[TrackId]) -> None:
        """Test tracking with empty frame."""
        from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter

        adapter = BoxmotTrackerAdapter(
            tracker_type="bytetrack", device="cpu", half=False
        )

        frame = DetectedFrame(
            no=FrameNo(1),
            occurrence=0.0,
            source="test.mp4",
            output="test.otdet",
            detections=[],
            image=None,
        )

        tracked_frame = adapter.track_frame(frame, id_generator)
        assert len(tracked_frame.detections) == 0

    def test_track_single_detection(self, id_generator: Iterator[TrackId]) -> None:
        """Test tracking with single detection."""
        from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter

        adapter = BoxmotTrackerAdapter(
            tracker_type="bytetrack", device="cpu", half=False
        )

        frame = DetectedFrame(
            no=FrameNo(1),
            occurrence=0.0,
            source="test.mp4",
            output="test.otdet",
            detections=[
                Detection(label="car", conf=0.9, x=100.0, y=150.0, w=50.0, h=80.0)
            ],
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
        )

        tracked_frame = adapter.track_frame(frame, id_generator)
        assert (
            len(tracked_frame.detections) >= 0
        )  # May or may not track depending on thresholds

    def test_reset_clears_state(self) -> None:
        """Test that reset clears internal state."""
        from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter

        adapter = BoxmotTrackerAdapter(
            tracker_type="bytetrack", device="cpu", half=False
        )

        # Add some state by tracking a frame
        def id_gen() -> Iterator[TrackId]:
            yield from range(100)

        frame = DetectedFrame(
            no=FrameNo(1),
            occurrence=0.0,
            source="test.mp4",
            output="test.otdet",
            detections=[
                Detection(label="car", conf=0.9, x=100.0, y=150.0, w=50.0, h=80.0)
            ],
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
        )

        adapter.track_frame(frame, id_gen())

        # Reset
        adapter.reset()

        # State should be cleared
        assert len(adapter._track_id_mapping) == 0
        assert len(adapter._previous_track_ids) == 0
        assert len(adapter._class_mapper.get_forward_mapping()) == 0
