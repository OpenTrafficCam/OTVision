"""Frame-by-frame association adapted from the original ByteTrack and BoT-SORT.

https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/byte_tracker.py
https://github.com/NirAharon/BoT-SORT/blob/main/tracker/bot_sort.py

MIT License

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

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.domain.detection import Detection, TrackedDetection, TrackId
from OTVision.domain.frame import DetectedFrame, TrackedFrame
from OTVision.track.model.tracking_interfaces import IdGenerator, Tracker
from OTVision.track.tracker.bot_sort.kalman_filter import (
    BOX_DIMENSIONS,
    STATE_BOX,
    STATE_VELOCITY,
    KalmanFilter,
)
from OTVision.track.tracker.bot_sort.matching import (
    BOX_SIZE,
    fuse_score,
    iou_distance,
    linear_assignment,
)

# Fixed matching costs from ByteTrack and BoT-SORT. Unlike
# TRACK.BOT_SORT.MATCH_THRESH, the reference implementations do not expose them.
_LOW_MATCH_THRESHOLD = 0.5
"""Highest cost (1 - IoU) for matching a low-confidence detection to a track
that the first pass left unmatched: the boxes need an IoU above 0.5."""
_UNCONFIRMED_MATCH_THRESHOLD = 0.7
"""Highest cost for matching a track that was born in the previous frame. With
TRACK.BOT_SORT.FUSE_SCORE the cost is 1 - IoU * confidence, so the pair needs
IoU * confidence above 0.3. Without it the boxes need an IoU above 0.3."""
_DUPLICATE_DISTANCE = 0.15
"""A tracked and a lost track closer than this cost (1 - IoU, so an IoU above
0.85) follow the same object, and the one with the shorter span is ended."""

_FIRST_FRAME = 1
"""Frame number after a reset. Tracks born in it are confirmed at once, because
no earlier frame could have confirmed them."""


@dataclass
class _Track:
    id: TrackId
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    first_frame: int
    last_frame: int

    @property
    def span(self) -> int:
        return self.last_frame - self.first_frame


def _box(detection: Detection) -> tuple[float, float, float, float]:
    return detection.x, detection.y, detection.w, detection.h


def _track_boxes(tracks: list[_Track]) -> NDArray[np.float64]:
    boxes = np.array([track.mean[STATE_BOX] for track in tracks])
    return boxes.reshape(-1, BOX_DIMENSIONS)


def _missed_frames(track: _Track, through_frame: int) -> int:
    """Return how many frames after its last match, up to and including
    `through_frame`, the track went without a detection."""
    return through_frame - track.last_frame


def _duplicates(tracked: list[_Track], lost: list[_Track]) -> dict[TrackId, _Track]:
    distances = iou_distance(_track_boxes(tracked), _track_boxes(lost))
    duplicates: dict[TrackId, _Track] = {}
    for tracked_index, lost_index in np.argwhere(distances < _DUPLICATE_DISTANCE):
        active, missing = tracked[tracked_index], lost[lost_index]
        shorter = missing if active.span > missing.span else active
        duplicates[shorter.id] = shorter
    return duplicates


class BotSortTracker(Tracker):
    """BoT-SORT with confidence-based association and a box motion model."""

    def __init__(self, get_current_config: GetCurrentConfig) -> None:
        self._get_current_config = get_current_config
        self._kalman = KalmanFilter()
        self._frame = 0
        self._tracked: list[_Track] = []
        self._lost: list[_Track] = []
        self._unconfirmed: list[_Track] = []

    def reset(self) -> None:
        self._frame = 0
        self._tracked.clear()
        self._lost.clear()
        self._unconfirmed.clear()

    def _associate(
        self,
        tracks: list[_Track],
        detections: list[Detection],
        threshold: float,
        use_score: bool,
        emitted: list[TrackedDetection],
    ) -> tuple[list[_Track], list[Detection]]:
        boxes = np.array([_box(d) for d in detections]).reshape(-1, BOX_DIMENSIONS)
        costs = iou_distance(_track_boxes(tracks), boxes)
        if use_score:
            costs = fuse_score(costs, np.array([d.conf for d in detections]))
        matches, unmatched_tracks, unmatched_detections = linear_assignment(
            costs, threshold
        )
        for track_index, detection_index in matches:
            track = tracks[track_index]
            detection = detections[detection_index]
            track.mean, track.covariance = self._kalman.update(
                track.mean, track.covariance, _box(detection)
            )
            track.last_frame = self._frame
            emitted.append(detection.of_track(track.id, is_first=False))
        return (
            [tracks[i] for i in unmatched_tracks],
            [detections[i] for i in unmatched_detections],
        )

    def track_frame(
        self, frame: DetectedFrame, id_generator: IdGenerator
    ) -> TrackedFrame:
        self._frame += 1
        previous_frame = self._frame - 1
        config = self._get_current_config.get().track.bot_sort
        high = [d for d in frame.detections if d.conf >= config.track_high_thresh]
        low = [
            d
            for d in frame.detections
            if config.track_low_thresh < d.conf < config.track_high_thresh
        ]
        emitted: list[TrackedDetection] = []
        # A config change may shorten the window while a track is already lost.
        expired = [
            track
            for track in self._lost
            if _missed_frames(track, previous_frame) > config.t_miss_max
        ]
        self._lost = [
            track
            for track in self._lost
            if _missed_frames(track, previous_frame) <= config.t_miss_max
        ]

        pool = self._tracked + self._lost
        if pool:
            means = np.stack([track.mean for track in pool])
            covariances = np.stack([track.covariance for track in pool])
            # Lost tracks keep moving, but stop growing or shrinking, as in BoT-SORT.
            lost_velocities = means[len(self._tracked) :, STATE_VELOCITY]
            lost_velocities[:, BOX_SIZE] = 0
            means, covariances = self._kalman.multi_predict(means, covariances)
            for track, mean, covariance in zip(pool, means, covariances):
                track.mean, track.covariance = mean, covariance

        unmatched, high = self._associate(
            pool, high, config.match_thresh, config.fuse_score, emitted
        )
        still_tracked = [
            track for track in unmatched if track.last_frame == previous_frame
        ]
        newly_lost, _ = self._associate(
            still_tracked, low, _LOW_MATCH_THRESHOLD, use_score=False, emitted=emitted
        )

        refound = [track for track in self._lost if track.last_frame == self._frame]
        self._lost = [
            track for track in self._lost if track.last_frame != self._frame
        ] + newly_lost
        self._tracked = [
            track for track in self._tracked if track.last_frame == self._frame
        ]
        discarded, high = self._associate(
            self._unconfirmed,
            high,
            _UNCONFIRMED_MATCH_THRESHOLD,
            config.fuse_score,
            emitted,
        )
        self._tracked.extend(
            track for track in self._unconfirmed if track.last_frame == self._frame
        )
        self._unconfirmed = []

        new_tracks = self._tracked if self._frame == _FIRST_FRAME else self._unconfirmed
        for detection in high:
            if detection.conf >= config.new_track_thresh:
                track_id = next(id_generator)
                mean, covariance = self._kalman.initiate(_box(detection))
                new_tracks.append(
                    _Track(track_id, mean, covariance, self._frame, self._frame)
                )
                emitted.append(detection.of_track(track_id, is_first=True))
        self._tracked.extend(refound)

        expired.extend(
            track
            for track in self._lost
            if _missed_frames(track, self._frame) > config.t_miss_max
        )
        ended_ids = {track.id for track in expired}
        self._lost = [track for track in self._lost if track.id not in ended_ids]
        duplicates = _duplicates(self._tracked + self._unconfirmed, self._lost)
        self._tracked = [track for track in self._tracked if track.id not in duplicates]
        self._unconfirmed = [
            track for track in self._unconfirmed if track.id not in duplicates
        ]
        self._lost = [track for track in self._lost if track.id not in duplicates]
        # A track born in this frame has only its suppressed first detection, so
        # its ID was never emitted and must not be reported.
        ended = expired + [
            track for track in duplicates.values() if track.first_frame != self._frame
        ]
        finished = {track.id for track in ended if track.span >= config.t_min}
        discarded_ids = {track.id for track in ended if track.id not in finished}
        discarded_ids.update(track.id for track in discarded)
        return TrackedFrame(
            no=frame.no,
            occurrence=frame.occurrence,
            source=frame.source,
            output=frame.output,
            image=frame.image,
            detections=[d for d in emitted if d.track_id not in duplicates],
            finished_tracks=finished,
            discarded_tracks=discarded_ids,
        )
