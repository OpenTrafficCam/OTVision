import numpy as np
import pytest

from OTVision.track.tracker.botsort.matching import (
    fuse_score,
    iou_distance,
    linear_assignment,
)


def test_iou_distance_uses_plain_box_areas() -> None:
    tracks = np.array([[0, 0, 2, 2], [3, 0, 2, 2]])
    detections = np.array([[0, 0, 2, 2], [1, 0, 2, 2], [10, 0, 2, 2]])

    np.testing.assert_allclose(
        iou_distance(tracks, detections), [[0.0, 2 / 3, 1.0], [1.0, 1.0, 1.0]]
    )


def test_fuse_score_applies_each_detection_score_to_its_column() -> None:
    iou_costs = np.array([[0, 0.5, 0], [1, 0, 0.25]])
    detection_scores = np.array([0.25, 0.75, 0])

    np.testing.assert_allclose(
        fuse_score(iou_costs, detection_scores), [[0.75, 0.625, 1], [1, 0.25, 1]]
    )


def test_linear_assignment_prices_unmatched_items_during_optimization() -> None:
    costs = np.array([[0.05, 0.4], [0.4, 0.8]])

    # A forced assignment chooses both off-diagonal pairs for 0.8, and both
    # survive filtering. Matching only (0, 0) and leaving two items unmatched
    # costs 0.05 + 0.25 + 0.25, so the thresholded solution chooses that pair.
    matches, unmatched_tracks, unmatched_detections = linear_assignment(costs, 0.5)

    np.testing.assert_array_equal(matches, [[0, 0]])
    np.testing.assert_array_equal(unmatched_tracks, [1])
    np.testing.assert_array_equal(unmatched_detections, [1])


def test_linear_assignment_reports_indices_for_rectangular_costs() -> None:
    costs = np.array(
        [[0.8, 0.6, 0.1, 0.8], [0.3, 0.7, 0.45, 0.8], [0.9, 0.9, 0.9, 0.9]]
    )

    matches, unmatched_tracks, unmatched_detections = linear_assignment(costs, 0.5)

    np.testing.assert_array_equal(matches, [[0, 2], [1, 0]])
    np.testing.assert_array_equal(unmatched_tracks, [2])
    np.testing.assert_array_equal(unmatched_detections, [1, 3])


@pytest.mark.parametrize(("track_count", "detection_count"), [(0, 2), (2, 0), (0, 0)])
def test_matching_handles_empty_inputs(track_count: int, detection_count: int) -> None:
    track_boxes = np.zeros((track_count, 4))
    detection_boxes = np.zeros((detection_count, 4))
    detection_scores = np.ones(detection_count)

    costs = iou_distance(track_boxes, detection_boxes)
    assert costs.shape == (track_count, detection_count)
    fused_costs = fuse_score(costs, detection_scores)
    assert fused_costs.shape == (track_count, detection_count)
    matches, unmatched_tracks, unmatched_detections = linear_assignment(
        fused_costs, 0.5
    )

    assert matches.shape == (0, 2)
    np.testing.assert_array_equal(unmatched_tracks, list(range(track_count)))
    np.testing.assert_array_equal(unmatched_detections, list(range(detection_count)))
