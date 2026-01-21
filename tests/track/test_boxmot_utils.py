"""Unit tests for BOXMOT utility functions."""

from typing import Iterator

import numpy as np

from OTVision.dataformat import ACTUAL_FPS, RECORDED_FPS, VIDEO
from OTVision.domain.detection import Detection, TrackId
from OTVision.track.boxmot_utils import (
    ClassLabelMapper,
    boxmot_tracks_to_detections,
    clamp_xyxy_to_frame,
    detection_to_xyxy,
    detections_to_boxmot_array,
    extract_fps_from_metadata,
    xyxy_to_xywh_center,
)


class TestCoordinateConversion:
    """Test coordinate conversion functions."""

    def test_detection_to_xyxy(self) -> None:
        """Test conversion from center xywh to corner xyxy format."""
        detection = Detection(
            label="car",
            conf=0.9,
            x=100.0,  # center x
            y=150.0,  # center y
            w=50.0,  # width
            h=80.0,  # height
        )

        x1, y1, x2, y2 = detection_to_xyxy(detection)

        assert x1 == 75.0  # 100 - 50/2
        assert y1 == 110.0  # 150 - 80/2
        assert x2 == 125.0  # 100 + 50/2
        assert y2 == 190.0  # 150 + 80/2

    def test_xyxy_to_xywh_center(self) -> None:
        """Test conversion from corner xyxy to center xywh format."""
        x, y, w, h = xyxy_to_xywh_center(75.0, 110.0, 125.0, 190.0)

        assert x == 100.0
        assert y == 150.0
        assert w == 50.0
        assert h == 80.0

    def test_round_trip_conversion(self) -> None:
        """Test that converting back and forth preserves values."""
        original = Detection(
            label="pedestrian", conf=0.85, x=200.0, y=300.0, w=40.0, h=120.0
        )

        # Convert to xyxy and back
        x1, y1, x2, y2 = detection_to_xyxy(original)
        x, y, w, h = xyxy_to_xywh_center(x1, y1, x2, y2)

        assert x == original.x
        assert y == original.y
        assert w == original.w
        assert h == original.h


class TestDetectionsToBoxmotArray:
    """Test conversion from OTVision Detections to BOXMOT array format."""

    def test_empty_detections(self) -> None:
        """Test with empty detection list."""
        result = detections_to_boxmot_array([], {})
        assert result.shape == (0, 6)

    def test_single_detection(self) -> None:
        """Test conversion of single detection."""
        detections = [
            Detection(label="car", conf=0.95, x=100.0, y=150.0, w=50.0, h=80.0)
        ]
        class_mapping = {"car": 0}

        result = detections_to_boxmot_array(detections, class_mapping)

        assert result.shape == (1, 6)
        assert result[0, 0] == 75.0  # x1
        assert result[0, 1] == 110.0  # y1
        assert result[0, 2] == 125.0  # x2
        assert result[0, 3] == 190.0  # y2
        assert result[0, 4] == 0.95  # conf
        assert result[0, 5] == 0  # class id

    def test_multiple_detections(self) -> None:
        """Test conversion of multiple detections with different classes."""
        detections = [
            Detection(label="car", conf=0.9, x=100.0, y=150.0, w=50.0, h=80.0),
            Detection(label="pedestrian", conf=0.85, x=200.0, y=300.0, w=40.0, h=120.0),
            Detection(label="bicycle", conf=0.88, x=150.0, y=200.0, w=30.0, h=60.0),
        ]
        class_mapping = {"car": 0, "pedestrian": 1, "bicycle": 2}

        result = detections_to_boxmot_array(detections, class_mapping)

        assert result.shape == (3, 6)
        # Check class IDs
        assert result[0, 5] == 0
        assert result[1, 5] == 1
        assert result[2, 5] == 2

    def test_unknown_class_defaults_to_zero(self) -> None:
        """Test that unknown class labels default to class ID 0."""
        detections = [
            Detection(label="unknown", conf=0.8, x=100.0, y=100.0, w=50.0, h=50.0)
        ]
        class_mapping = {"car": 1}  # "unknown" not in mapping

        result = detections_to_boxmot_array(detections, class_mapping)

        assert result[0, 5] == 0  # Should default to 0


class TestBoxmotTracksToDetections:
    """Test conversion from BOXMOT output to OTVision TrackedDetections."""

    def test_empty_tracks(self) -> None:
        """Test with empty tracks array."""
        empty_tracks = np.empty((0, 8), dtype=np.float32)

        def mock_id_gen() -> Iterator[TrackId]:
            counter = 0
            while True:
                counter += 1
                yield counter

        detections, current_ids, mapping = boxmot_tracks_to_detections(
            empty_tracks, {}, {}, mock_id_gen(), set()
        )

        assert len(detections) == 0
        assert len(current_ids) == 0
        assert len(mapping) == 0

    def test_single_new_track(self) -> None:
        """Test conversion of single new track."""
        # BOXMOT output: [x1, y1, x2, y2, id, conf, cls, ind]
        tracks = np.array(
            [[75.0, 110.0, 125.0, 190.0, 1, 0.95, 0, 0]], dtype=np.float32
        )

        def id_gen() -> Iterator[TrackId]:
            counter = 0
            while True:
                counter += 1
                yield counter

        class_mapping_reverse = {0: "car"}

        detections, current_ids, mapping = boxmot_tracks_to_detections(
            tracks, class_mapping_reverse, {}, id_gen(), set()
        )

        assert len(detections) == 1
        assert detections[0].label == "car"
        assert abs(detections[0].conf - 0.95) < 0.0001
        assert detections[0].x == 100.0  # center x
        assert detections[0].y == 150.0  # center y
        assert detections[0].w == 50.0
        assert detections[0].h == 80.0
        assert detections[0].is_first is True
        assert current_ids == {1}
        assert 1 in mapping

    def test_existing_track_continuation(self) -> None:
        """Test that existing tracks are not marked as first."""
        tracks = np.array(
            [[75.0, 110.0, 125.0, 190.0, 1, 0.95, 0, 0]], dtype=np.float32
        )

        existing_mapping = {1: 999}

        def id_gen() -> Iterator[TrackId]:
            while True:
                yield -1  # Should not be called

        class_mapping_reverse = {0: "car"}

        detections, current_ids, mapping = boxmot_tracks_to_detections(
            tracks, class_mapping_reverse, existing_mapping, id_gen(), {1}
        )

        assert len(detections) == 1
        assert detections[0].is_first is False
        assert detections[0].track_id == 999
        assert mapping == existing_mapping  # Mapping unchanged

    def test_multiple_tracks(self) -> None:
        """Test conversion of multiple tracks."""
        # Two new tracks
        tracks = np.array(
            [
                [75.0, 110.0, 125.0, 190.0, 1, 0.95, 0, 0],
                [180.0, 240.0, 220.0, 360.0, 2, 0.85, 1, 1],
            ],
            dtype=np.float32,
        )

        def id_gen() -> Iterator[TrackId]:
            counter = 0
            while True:
                counter += 1
                yield counter

        class_mapping_reverse = {0: "car", 1: "pedestrian"}

        detections, current_ids, mapping = boxmot_tracks_to_detections(
            tracks, class_mapping_reverse, {}, id_gen(), set()
        )

        assert len(detections) == 2
        assert detections[0].label == "car"
        assert detections[1].label == "pedestrian"
        assert current_ids == {1, 2}
        assert len(mapping) == 2


class TestClassLabelMapper:
    """Test the ClassLabelMapper utility."""

    def test_initial_state(self) -> None:
        """Test mapper starts empty."""
        mapper = ClassLabelMapper()
        assert mapper.get_forward_mapping() == {}
        assert mapper.get_reverse_mapping() == {}

    def test_get_id_creates_mapping(self) -> None:
        """Test that get_id creates new mappings."""
        mapper = ClassLabelMapper()

        car_id = mapper.get_id("car")
        assert car_id == 0

        pedestrian_id = mapper.get_id("pedestrian")
        assert pedestrian_id == 1

        # Getting same label returns same ID
        assert mapper.get_id("car") == 0

    def test_get_label(self) -> None:
        """Test retrieving labels by ID."""
        mapper = ClassLabelMapper()
        mapper.get_id("car")
        mapper.get_id("pedestrian")

        assert mapper.get_label(0) == "car"
        assert mapper.get_label(1) == "pedestrian"

    def test_get_label_unknown_id(self) -> None:
        """Test that unknown IDs return formatted string."""
        mapper = ClassLabelMapper()
        assert mapper.get_label(99) == "class_99"

    def test_bidirectional_mappings(self) -> None:
        """Test that forward and reverse mappings are consistent."""
        mapper = ClassLabelMapper()
        labels = ["car", "pedestrian", "bicycle", "truck"]

        for label in labels:
            mapper.get_id(label)

        forward = mapper.get_forward_mapping()
        reverse = mapper.get_reverse_mapping()

        for label, class_id in forward.items():
            assert reverse[class_id] == label


class TestClampXyxyToFrame:
    """Test the clamp_xyxy_to_frame function."""

    def test_no_clamping_needed(self) -> None:
        """Test that valid coordinates within frame are unchanged."""
        result = clamp_xyxy_to_frame(100.0, 100.0, 200.0, 200.0, 1920, 1080)
        assert result == (100.0, 100.0, 200.0, 200.0)

    def test_clamp_negative_x1(self) -> None:
        """Test clamping negative x1 coordinate."""
        result = clamp_xyxy_to_frame(-50.0, 100.0, 100.0, 200.0, 1920, 1080)
        assert result is not None
        assert result[0] == 0.0  # x1 clamped
        assert result[1] == 100.0  # y1 unchanged
        assert result[2] == 100.0  # x2 unchanged
        assert result[3] == 200.0  # y2 unchanged

    def test_clamp_negative_y1(self) -> None:
        """Test clamping negative y1 coordinate."""
        result = clamp_xyxy_to_frame(100.0, -30.0, 200.0, 200.0, 1920, 1080)
        assert result is not None
        assert result[0] == 100.0  # x1 unchanged
        assert result[1] == 0.0  # y1 clamped
        assert result[2] == 200.0  # x2 unchanged
        assert result[3] == 200.0  # y2 unchanged

    def test_clamp_x2_beyond_frame(self) -> None:
        """Test clamping x2 beyond frame width."""
        result = clamp_xyxy_to_frame(1800.0, 100.0, 2000.0, 200.0, 1920, 1080)
        assert result is not None
        assert result[0] == 1800.0  # x1 unchanged
        assert result[1] == 100.0  # y1 unchanged
        assert result[2] == 1920.0  # x2 clamped to frame width
        assert result[3] == 200.0  # y2 unchanged

    def test_clamp_y2_beyond_frame(self) -> None:
        """Test clamping y2 beyond frame height."""
        result = clamp_xyxy_to_frame(100.0, 1000.0, 200.0, 1200.0, 1920, 1080)
        assert result is not None
        assert result[0] == 100.0  # x1 unchanged
        assert result[1] == 1000.0  # y1 unchanged
        assert result[2] == 200.0  # x2 unchanged
        assert result[3] == 1080.0  # y2 clamped to frame height

    def test_clamp_all_coordinates(self) -> None:
        """Test clamping all coordinates that extend beyond frame."""
        result = clamp_xyxy_to_frame(-50.0, -30.0, 2000.0, 1200.0, 1920, 1080)
        assert result is not None
        assert result[0] == 0.0  # x1 clamped
        assert result[1] == 0.0  # y1 clamped
        assert result[2] == 1920.0  # x2 clamped
        assert result[3] == 1080.0  # y2 clamped

    def test_filter_completely_outside_left(self) -> None:
        """Test that bbox completely to the left of frame is filtered."""
        result = clamp_xyxy_to_frame(-200.0, 100.0, -50.0, 200.0, 1920, 1080)
        assert result is None

    def test_filter_completely_outside_right(self) -> None:
        """Test that bbox completely to the right of frame is filtered."""
        result = clamp_xyxy_to_frame(2000.0, 100.0, 2200.0, 200.0, 1920, 1080)
        assert result is None

    def test_filter_completely_outside_top(self) -> None:
        """Test that bbox completely above frame is filtered."""
        result = clamp_xyxy_to_frame(100.0, -200.0, 200.0, -50.0, 1920, 1080)
        assert result is None

    def test_filter_completely_outside_bottom(self) -> None:
        """Test that bbox completely below frame is filtered."""
        result = clamp_xyxy_to_frame(100.0, 1200.0, 200.0, 1400.0, 1920, 1080)
        assert result is None

    def test_filter_zero_width_after_clamping(self) -> None:
        """Test that bbox with zero width after clamping is filtered."""
        # x1 and x2 both clamp to 0 when entirely negative
        result = clamp_xyxy_to_frame(-100.0, 100.0, -50.0, 200.0, 1920, 1080)
        assert result is None

    def test_filter_zero_height_after_clamping(self) -> None:
        """Test that bbox with zero height after clamping is filtered."""
        # y1 and y2 both clamp to 0 when entirely negative
        result = clamp_xyxy_to_frame(100.0, -100.0, 200.0, -50.0, 1920, 1080)
        assert result is None

    def test_edge_case_touching_left_edge(self) -> None:
        """Test bbox that starts exactly at x=0."""
        result = clamp_xyxy_to_frame(0.0, 100.0, 100.0, 200.0, 1920, 1080)
        assert result == (0.0, 100.0, 100.0, 200.0)

    def test_edge_case_touching_right_edge(self) -> None:
        """Test bbox that ends exactly at frame width."""
        result = clamp_xyxy_to_frame(1820.0, 100.0, 1920.0, 200.0, 1920, 1080)
        assert result == (1820.0, 100.0, 1920.0, 200.0)


class TestBoxmotTracksToDetectionsWithClamping:
    """Test boxmot_tracks_to_detections with coordinate clamping."""

    def test_negative_coordinates_get_clamped(self) -> None:
        """Test that tracks with negative coordinates are clamped correctly."""
        # Track with x1=-50 (should clamp to 0)
        # xyxy: (-50, 100, 100, 200) -> after clamping: (0, 100, 100, 200)
        # center: x = 50, y = 150, w = 100, h = 100
        tracks = np.array(
            [[-50.0, 100.0, 100.0, 200.0, 1, 0.95, 0, 0]], dtype=np.float32
        )

        def id_gen() -> Iterator[TrackId]:
            counter = 0
            while True:
                counter += 1
                yield counter

        class_mapping_reverse = {0: "car"}

        detections, current_ids, mapping = boxmot_tracks_to_detections(
            tracks,
            class_mapping_reverse,
            {},
            id_gen(),
            set(),
            frame_width=1920,
            frame_height=1080,
        )

        assert len(detections) == 1
        assert detections[0].x == 50.0  # center x after clamping: (0 + 100) / 2
        assert detections[0].y == 150.0  # center y: (100 + 200) / 2
        assert detections[0].w == 100.0  # width: 100 - 0
        assert detections[0].h == 100.0  # height: 200 - 100

    def test_out_of_frame_tracks_filtered(self) -> None:
        """Test that tracks completely outside frame are filtered out."""
        # Two tracks: one inside frame, one completely outside
        tracks = np.array(
            [
                [100.0, 100.0, 200.0, 200.0, 1, 0.95, 0, 0],  # Inside frame
                [2000.0, 100.0, 2200.0, 200.0, 2, 0.90, 0, 0],  # Outside frame
            ],
            dtype=np.float32,
        )

        def id_gen() -> Iterator[TrackId]:
            counter = 0
            while True:
                counter += 1
                yield counter

        class_mapping_reverse = {0: "car"}

        detections, current_ids, mapping = boxmot_tracks_to_detections(
            tracks,
            class_mapping_reverse,
            {},
            id_gen(),
            set(),
            frame_width=1920,
            frame_height=1080,
        )

        # Only the inside-frame track should be present
        assert len(detections) == 1
        assert detections[0].x == 150.0  # center of first track
        assert current_ids == {1}  # Only track ID 1

    def test_backward_compatibility_no_dimensions(self) -> None:
        """Test that no clamping occurs when dimensions not provided."""
        # Track with negative coordinates (would be invalid but not clamped)
        tracks = np.array(
            [[-50.0, -30.0, 100.0, 200.0, 1, 0.95, 0, 0]], dtype=np.float32
        )

        def id_gen() -> Iterator[TrackId]:
            counter = 0
            while True:
                counter += 1
                yield counter

        class_mapping_reverse = {0: "car"}

        # Call without frame dimensions (backward compatible)
        detections, current_ids, mapping = boxmot_tracks_to_detections(
            tracks,
            class_mapping_reverse,
            {},
            id_gen(),
            set(),
        )

        assert len(detections) == 1
        # Center should be calculated from original (negative) coordinates
        # x = (-50 + 100) / 2 = 25
        # y = (-30 + 200) / 2 = 85
        assert detections[0].x == 25.0
        assert detections[0].y == 85.0

    def test_clamped_track_dimensions(self) -> None:
        """Test that clamped track has correct reduced dimensions."""
        # Track extending past right edge: x2 = 2000 should clamp to 1920
        # xyxy: (1800, 100, 2000, 200) -> clamped: (1800, 100, 1920, 200)
        tracks = np.array(
            [[1800.0, 100.0, 2000.0, 200.0, 1, 0.95, 0, 0]], dtype=np.float32
        )

        def id_gen() -> Iterator[TrackId]:
            counter = 0
            while True:
                counter += 1
                yield counter

        class_mapping_reverse = {0: "car"}

        detections, current_ids, mapping = boxmot_tracks_to_detections(
            tracks,
            class_mapping_reverse,
            {},
            id_gen(),
            set(),
            frame_width=1920,
            frame_height=1080,
        )

        assert len(detections) == 1
        # After clamping: x1=1800, x2=1920, y1=100, y2=200
        assert detections[0].x == 1860.0  # center x: (1800 + 1920) / 2
        assert detections[0].w == 120.0  # width: 1920 - 1800

    def test_multiple_tracks_with_clamping(self) -> None:
        """Test clamping with multiple tracks, some filtered, some clamped."""
        tracks = np.array(
            [
                [-100.0, 100.0, 50.0, 200.0, 1, 0.95, 0, 0],  # Clamp x1 to 0
                [100.0, 100.0, 200.0, 200.0, 2, 0.90, 0, 0],  # No clamping needed
                [3000.0, 100.0, 3200.0, 200.0, 3, 0.85, 0, 0],  # Completely outside
            ],
            dtype=np.float32,
        )

        def id_gen() -> Iterator[TrackId]:
            counter = 0
            while True:
                counter += 1
                yield counter

        class_mapping_reverse = {0: "car"}

        detections, current_ids, mapping = boxmot_tracks_to_detections(
            tracks,
            class_mapping_reverse,
            {},
            id_gen(),
            set(),
            frame_width=1920,
            frame_height=1080,
        )

        # Two tracks should remain (track 3 is completely outside)
        assert len(detections) == 2
        # First track clamped: x1=0, x2=50 -> center x = 25, w = 50
        assert detections[0].x == 25.0
        assert detections[0].w == 50.0
        # Second track unchanged: x1=100, x2=200 -> center x = 150, w = 100
        assert detections[1].x == 150.0
        assert detections[1].w == 100.0
        # Only track IDs 1 and 2 should be in current_ids
        assert current_ids == {1, 2}


class TestExtractFpsFromMetadata:
    """Test the extract_fps_from_metadata function."""

    def test_extract_actual_fps(self) -> None:
        """Test extraction of actual_fps from metadata."""
        metadata = {VIDEO: {ACTUAL_FPS: 25.0}}

        result = extract_fps_from_metadata(metadata)

        assert result == 25.0

    def test_extract_recorded_fps_fallback(self) -> None:
        """Test fallback to recorded_fps when actual_fps not present."""
        metadata = {VIDEO: {RECORDED_FPS: 30.0}}

        result = extract_fps_from_metadata(metadata)

        assert result == 30.0

    def test_actual_fps_takes_precedence(self) -> None:
        """Test that actual_fps takes precedence over recorded_fps."""
        metadata = {VIDEO: {ACTUAL_FPS: 25.0, RECORDED_FPS: 30.0}}

        result = extract_fps_from_metadata(metadata)

        assert result == 25.0

    def test_returns_none_when_no_fps_available(self) -> None:
        """Test that None is returned when no FPS data is available."""
        metadata: dict[str, dict] = {VIDEO: {}}

        result = extract_fps_from_metadata(metadata)

        assert result is None

    def test_returns_none_when_video_key_missing(self) -> None:
        """Test that None is returned when video key is missing."""
        metadata = {"other_key": "value"}

        result = extract_fps_from_metadata(metadata)

        assert result is None

    def test_returns_none_for_empty_metadata(self) -> None:
        """Test that None is returned for empty metadata."""
        metadata: dict = {}

        result = extract_fps_from_metadata(metadata)

        assert result is None

    def test_ignores_zero_actual_fps(self) -> None:
        """Test that zero actual_fps is ignored, falls back to recorded_fps."""
        metadata = {VIDEO: {ACTUAL_FPS: 0, RECORDED_FPS: 30.0}}

        result = extract_fps_from_metadata(metadata)

        assert result == 30.0

    def test_ignores_negative_actual_fps(self) -> None:
        """Test that negative actual_fps is ignored, falls back to recorded_fps."""
        metadata = {VIDEO: {ACTUAL_FPS: -1, RECORDED_FPS: 30.0}}

        result = extract_fps_from_metadata(metadata)

        assert result == 30.0

    def test_ignores_none_actual_fps(self) -> None:
        """Test that None actual_fps is ignored, falls back to recorded_fps."""
        metadata = {VIDEO: {ACTUAL_FPS: None, RECORDED_FPS: 30.0}}

        result = extract_fps_from_metadata(metadata)

        assert result == 30.0

    def test_returns_none_when_both_fps_values_invalid(self) -> None:
        """Test that None is returned when both FPS values are invalid."""
        metadata = {VIDEO: {ACTUAL_FPS: 0, RECORDED_FPS: -5}}

        result = extract_fps_from_metadata(metadata)

        assert result is None

    def test_converts_integer_to_float(self) -> None:
        """Test that integer FPS values are converted to float."""
        metadata = {VIDEO: {ACTUAL_FPS: 30}}

        result = extract_fps_from_metadata(metadata)

        assert result == 30.0
        assert isinstance(result, float)
