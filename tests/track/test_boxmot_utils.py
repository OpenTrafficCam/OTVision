"""Unit tests for BOXMOT utility functions."""

from typing import Iterator

import numpy as np

from OTVision.domain.detection import Detection, TrackId
from OTVision.track.boxmot_utils import (
    ClassLabelMapper,
    boxmot_tracks_to_detections,
    detection_to_xyxy,
    detections_to_boxmot_array,
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
