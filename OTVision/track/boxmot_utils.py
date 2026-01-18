"""Utility functions for BOXMOT integration.

This module provides conversion utilities between OTVision's detection format
and BOXMOT's expected input/output formats.
"""

from typing import Iterator

import numpy as np
from numpy import ndarray

from OTVision.domain.detection import Detection, TrackId, TrackedDetection

IdGenerator = Iterator[TrackId]


def detection_to_xyxy(detection: Detection) -> tuple[float, float, float, float]:
    """Convert OTVision Detection (xywh center format) to xyxy corner format.

    Args:
        detection: Detection with center coordinates (x, y) and dimensions (w, h)

    Returns:
        Tuple of (x1, y1, x2, y2) representing top-left and bottom-right corners
    """
    x1 = detection.x - detection.w / 2
    y1 = detection.y - detection.h / 2
    x2 = detection.x + detection.w / 2
    y2 = detection.y + detection.h / 2
    return (x1, y1, x2, y2)


def xyxy_to_xywh_center(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float, float]:
    """Convert xyxy corner format to xywh center format.

    Args:
        x1: Left x coordinate
        y1: Top y coordinate
        x2: Right x coordinate
        y2: Bottom y coordinate

    Returns:
        Tuple of (x, y, w, h) where x,y are center coordinates
    """
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2
    y = y1 + h / 2
    return (x, y, w, h)


def detections_to_boxmot_array(
    detections: list[Detection], class_mapping: dict[str, int]
) -> ndarray:
    """Convert OTVision Detections to BOXMOT input array format.

    Args:
        detections: List of OTVision Detection objects
        class_mapping: Mapping from class label strings to numeric IDs

    Returns:
        NumPy array of shape (N, 6) with columns [x1, y1, x2, y2, conf, cls]
    """
    if not detections:
        return np.empty((0, 6), dtype=np.float32)

    boxmot_dets = []
    for det in detections:
        x1, y1, x2, y2 = detection_to_xyxy(det)
        cls_id = class_mapping.get(det.label, 0)
        boxmot_dets.append([x1, y1, x2, y2, det.conf, cls_id])

    return np.array(boxmot_dets, dtype=np.float32)


def boxmot_tracks_to_detections(
    tracks: ndarray,
    class_mapping_reverse: dict[int, str],
    track_id_mapping: dict[int, TrackId],
    id_generator: IdGenerator,
    previous_track_ids: set[int],
) -> tuple[list[TrackedDetection], set[int], dict[int, TrackId]]:
    """Convert BOXMOT output array to OTVision TrackedDetections.

    Args:
        tracks: BOXMOT output array of shape (M, 8) with columns
            [x1, y1, x2, y2, id, conf, cls, ind]
        class_mapping_reverse: Mapping from numeric IDs to class label strings
        track_id_mapping: Current mapping from BOXMOT IDs to OTVision TrackIds
        id_generator: Generator for new track IDs
        previous_track_ids: Set of BOXMOT track IDs from previous frame

    Returns:
        Tuple of (tracked_detections, current_track_ids, updated_mapping)
        where current_track_ids is the set of BOXMOT IDs in this frame
    """
    tracked_detections: list[TrackedDetection] = []
    current_track_ids: set[int] = set()

    if len(tracks) == 0:
        return (tracked_detections, current_track_ids, track_id_mapping)

    for track in tracks:
        x1, y1, x2, y2, boxmot_id, conf, cls_id, _ = track
        boxmot_id = int(boxmot_id)
        cls_id = int(cls_id)

        # Convert to center coordinates
        x, y, w, h = xyxy_to_xywh_center(x1, y1, x2, y2)

        # Get or create OTVision track ID
        if boxmot_id not in track_id_mapping:
            track_id = next(id_generator)
            track_id_mapping[boxmot_id] = track_id
            is_first = True
        else:
            track_id = track_id_mapping[boxmot_id]
            is_first = False

        # Get class label
        label = class_mapping_reverse.get(cls_id, f"class_{cls_id}")

        tracked_detection = TrackedDetection(
            label=label,
            conf=float(conf),
            x=float(x),
            y=float(y),
            w=float(w),
            h=float(h),
            is_first=is_first,
            track_id=track_id,
        )
        tracked_detections.append(tracked_detection)
        current_track_ids.add(boxmot_id)

    return (tracked_detections, current_track_ids, track_id_mapping)


class ClassLabelMapper:
    """Bidirectional mapper between string class labels and numeric IDs."""

    def __init__(self) -> None:
        self._label_to_id: dict[str, int] = {}
        self._id_to_label: dict[int, str] = {}
        self._next_id: int = 0

    def get_id(self, label: str) -> int:
        """Get numeric ID for a class label, creating new mapping if needed.

        Args:
            label: Class label string

        Returns:
            Numeric class ID
        """
        if label not in self._label_to_id:
            self._label_to_id[label] = self._next_id
            self._id_to_label[self._next_id] = label
            self._next_id += 1
        return self._label_to_id[label]

    def get_label(self, class_id: int) -> str:
        """Get class label for a numeric ID.

        Args:
            class_id: Numeric class ID

        Returns:
            Class label string, or f"class_{class_id}" if unknown
        """
        return self._id_to_label.get(class_id, f"class_{class_id}")

    def get_forward_mapping(self) -> dict[str, int]:
        """Get complete label-to-ID mapping.

        Returns:
            Dictionary mapping class labels to numeric IDs
        """
        return self._label_to_id.copy()

    def get_reverse_mapping(self) -> dict[int, str]:
        """Get complete ID-to-label mapping.

        Returns:
            Dictionary mapping numeric IDs to class labels
        """
        return self._id_to_label.copy()
