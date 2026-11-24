"""Numeric association helpers for BoT-SORT.

Adapted from the MIT-licensed ByteTrack and BoT-SORT matching modules:
https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/matching.py
https://github.com/NirAharon/BoT-SORT/blob/main/tracker/matching.py

Copyright (c) 2021 Yifu Zhang
Copyright (c) 2022 Nir Aharon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


def iou_distance(
    track_boxes: NDArray[np.integer | np.floating],
    detection_boxes: NDArray[np.integer | np.floating],
) -> NDArray[np.float64]:
    """Return pairwise 1 - IoU for (center_x, center_y, width, height) boxes."""
    tracks = track_boxes.astype(np.float64, copy=False)
    detections = detection_boxes.astype(np.float64, copy=False)
    track_half_size = tracks[:, 2:] / 2
    detection_half_size = detections[:, 2:] / 2
    left_top = np.maximum(
        (tracks[:, :2] - track_half_size)[:, None, :],
        detections[:, :2] - detection_half_size,
    )
    right_bottom = np.minimum(
        (tracks[:, :2] + track_half_size)[:, None, :],
        detections[:, :2] + detection_half_size,
    )
    intersection = np.maximum(right_bottom - left_top, 0).prod(axis=2)
    track_area = tracks[:, 2:].prod(axis=1)
    detection_area = detections[:, 2:].prod(axis=1)
    union = track_area[:, None] + detection_area - intersection
    iou = np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )
    return 1 - iou


def fuse_score(
    iou_costs: NDArray[np.integer | np.floating],
    detection_scores: NDArray[np.integer | np.floating],
) -> NDArray[np.float64]:
    """Weight IoU by the confidence of the detection in each column."""
    iou_similarity = 1 - iou_costs.astype(np.float64, copy=False)
    scores = detection_scores.astype(np.float64, copy=False)
    return 1 - iou_similarity * scores


def linear_assignment(
    costs: NDArray[np.integer | np.floating], threshold: float
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Return matched (track, detection) pairs and unmatched indices for each side.

    Rows index tracks and columns index detections. Each unmatched item costs
    half the threshold, so a pair at exactly the threshold ties with leaving
    both items unmatched.
    """
    track_count, detection_count = costs.shape
    if costs.size == 0:
        return (
            np.empty((0, 2), dtype=np.intp),
            np.arange(track_count, dtype=np.intp),
            np.arange(detection_count, dtype=np.intp),
        )

    size = track_count + detection_count
    extended_costs = np.full((size, size), threshold / 2, dtype=np.float64)
    extended_costs[:track_count, :detection_count] = costs
    extended_costs[track_count:, detection_count:] = 0

    rows, columns = linear_sum_assignment(extended_costs)
    real_rows = rows < track_count
    real_columns = columns < detection_count
    matched = real_rows & real_columns
    matches = np.column_stack((rows[matched], columns[matched]))
    unmatched_tracks = rows[real_rows & ~real_columns]
    unmatched_detections = np.sort(columns[~real_rows & real_columns])
    return matches, unmatched_tracks, unmatched_detections
