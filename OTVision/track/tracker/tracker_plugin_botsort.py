"""
OTVision BoT-SORT tracker adapter using ultralytics BOTSORT.
"""

# Copyright (C) 2022 OpenTrafficCam Contributors
# <https://github.com/OpenTrafficCam
# <team@opentrafficcam.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import logging
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from OTVision import dataformat
from OTVision.application.config import TrackConfig
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.application.track.botsort_params import (
    extract_frame_rate_from_metadata,
    resolve_botsort_tracker_params,
    to_frame_rate,
    validate_botsort_gmc_config,
    validate_botsort_reid_config,
)
from OTVision.domain.detection import TrackId
from OTVision.domain.frame import DetectedFrame, FrameNo, TrackedFrame
from OTVision.domain.tracker import TrackerLifecycle
from OTVision.helpers.files import read_json_bz2_metadata
from OTVision.helpers.log import LOGGER_NAME
from OTVision.track.model.tracking_interfaces import IdGenerator, Tracker

log = logging.getLogger(LOGGER_NAME)

NumpyIndex = int | slice | NDArray[np.bool_] | NDArray[np.integer]

# Ultralytics BOTSORT.update() (8.3.159) returns Nx8 float rows:
# [x1, y1, x2, y2, track_id, score, cls, det_idx]
_BOTSORT_UPDATE_COLS = 8
_BOTSORT_COL_TRACK_ID = 4
_BOTSORT_COL_DET_IDX = 7


def _xywh_center_to_xyxy(xywh: NDArray[np.floating]) -> NDArray[np.float32]:
    """Map center-format ``(x, y, w, h)`` boxes to ``(x1, y1, x2, y2)``.

    Matches :func:`ultralytics.utils.ops.xywh2xyxy` (used by BYTETracker / BoT-SORT).

    Args:
        xywh (NDArray[np.floating]): Boxes in center format.

    Returns:
        NDArray[np.float32]: Boxes in corner format.
    """
    if xywh.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x = np.asarray(xywh, dtype=np.float32)
    xy = x[..., :2]
    wh = x[..., 2:] / 2.0
    out = np.empty_like(x)
    out[..., :2] = xy - wh
    out[..., 2:] = xy + wh
    return out


def validate_botsort_update_rows(tracks: NDArray[np.floating] | None) -> None:
    """Fail loudly if Ultralytics ``BOTSORT.update()`` row layout changed.

    The pinned 8.3.159 contract is exactly eight columns. Wider layouts are
    rejected so a new trailing column cannot silently become ``det_idx``.

    Args:
        tracks (NDArray[np.floating] | None): Result of ``BOTSORT.update()``.

    Raises:
        ValueError: If the array shape is not ``Nx8``.
    """
    if tracks is None or len(tracks) == 0:
        return
    arr = np.asarray(tracks)
    if arr.ndim != 2 or arr.shape[1] != _BOTSORT_UPDATE_COLS:
        raise ValueError(
            "Unexpected BoT-SORT update() result shape "
            f"{getattr(arr, 'shape', type(arr))}; "
            f"expected Nx{_BOTSORT_UPDATE_COLS} rows "
            "[x1, y1, x2, y2, track_id, score, cls, det_idx]."
        )


class UltralyticsResultsLike(Protocol):
    """Subset of ultralytics ``Results`` API required by ``BYTETracker.update``."""

    conf: NDArray[np.floating]
    xywh: NDArray[np.floating]
    cls: NDArray[np.integer]

    @property
    def xyxy(self) -> NDArray[np.floating]:
        """Corner-format boxes used when ``img is not None`` (GMC)."""
        ...

    def __len__(self) -> int:
        """Number of detections in this result batch."""
        ...

    def __getitem__(self, item: NumpyIndex) -> UltralyticsResultsLike:
        """Boolean mask indexing as in ``results = results[remain_inds]``."""
        ...


class BoTSORTTrackerLike(Protocol):
    """Protocol for ultralytics BYTETracker/BOTSORT (8.3+)."""

    def update(
        self,
        results: UltralyticsResultsLike,
        img: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.floating]:
        """Associate detections and return Nx8 track rows."""
        ...


@dataclass(frozen=True)
class TrackAssignment:
    """The OTVision track a detection in the current frame was assigned to.

    Attributes:
        ot_id: OTVision track id the detection belongs to.
        is_first: Whether this frame is the first the track was ever seen in.
    """

    ot_id: TrackId
    is_first: bool


@dataclass
class _TrackEntry:
    """Lifecycle state of one track, keyed by its Ultralytics track id."""

    ot_id: TrackId
    first_frame: FrameNo
    last_frame: FrameNo
    age_missing: int = 0

    @property
    def frame_span(self) -> int:
        """Number of frames between the first and last observation.

        Returns:
            int: Frame span of the track.
        """
        return self.last_frame - self.first_frame


class TrackRegistry:
    """Owns the mapping from Ultralytics track ids to OTVision tracks.

    Keyed by Ultralytics track id, so eviction needs no reverse lookup. The
    registry also owns the lifecycle: which tracks were observed, how long each
    has been missing, and when a track is finished or discarded.
    """

    def __init__(self) -> None:
        self._entries: dict[int, _TrackEntry] = {}

    def clear(self) -> None:
        """Forget every track, for use at video-group boundaries."""
        self._entries.clear()

    def observe(
        self,
        botsort_track_id: int,
        frame_no: FrameNo,
        id_generator: IdGenerator,
    ) -> TrackAssignment:
        """Record that ``botsort_track_id`` was seen in frame ``frame_no``.

        Assigns a fresh OTVision track id the first time an Ultralytics id is
        seen, and resets the missing-frame counter on every hit.

        Args:
            botsort_track_id (int): Ultralytics track id from ``update()``.
            frame_no (FrameNo): Current frame number.
            id_generator (IdGenerator): Provider of new OTVision track ids.

        Returns:
            TrackAssignment: Assigned OTVision track and first-sighting flag.
        """
        entry = self._entries.get(botsort_track_id)
        if entry is None:
            entry = _TrackEntry(
                ot_id=next(id_generator),
                first_frame=frame_no,
                last_frame=frame_no,
            )
            self._entries[botsort_track_id] = entry
            return TrackAssignment(ot_id=entry.ot_id, is_first=True)

        entry.last_frame = frame_no
        entry.age_missing = 0
        return TrackAssignment(
            ot_id=entry.ot_id, is_first=entry.first_frame == frame_no
        )

    def age_unobserved(self, observed_botsort_track_ids: set[int]) -> None:
        """Increment the missing-frame counter of every unobserved track.

        Args:
            observed_botsort_track_ids (set[int]): Ids seen in the current frame.
        """
        for botsort_track_id, entry in self._entries.items():
            if botsort_track_id not in observed_botsort_track_ids:
                entry.age_missing += 1

    def evict_expired(
        self, lifecycle: TrackerLifecycle
    ) -> tuple[set[TrackId], set[TrackId]]:
        """Remove tracks missing for longer than ``t_miss_max``.

        A track is finished when it spans at least ``t_min`` frames, and
        discarded otherwise. Unlike the IOU tracker there is no ``sigma_h``
        gate: BoT-SORT already filters detections during association via
        ``track_high_thresh``, ``track_low_thresh`` and ``new_track_thresh``,
        so only sufficiently confident detections ever form a track.

        Args:
            lifecycle (TrackerLifecycle): Thresholds of the BoT-SORT config.

        Returns:
            tuple[set[TrackId], set[TrackId]]: Finished and discarded track ids.
        """
        finished: set[TrackId] = set()
        discarded: set[TrackId] = set()
        expired = [
            botsort_track_id
            for botsort_track_id, entry in self._entries.items()
            # Matches IOU-tracker semantics: finish only once a track has been
            # missing for MORE than t_miss_max consecutive frames. The IOU
            # tracker checks before incrementing; we increment first, so `<=`
            # here is equivalent to its `<`.
            if entry.age_missing > lifecycle.t_miss_max
        ]
        for botsort_track_id in expired:
            entry = self._entries.pop(botsort_track_id)
            if entry.frame_span >= lifecycle.t_min:
                finished.add(entry.ot_id)
            else:
                discarded.add(entry.ot_id)
        return finished, discarded


@dataclass
class UltralyticsResultsLite:
    """Minimal stand-in for ultralytics detection results.

    Supports ``conf``/``xywh``/``cls`` fields plus slicing.
    """

    conf: NDArray[np.floating]
    xywh: NDArray[np.floating]
    cls: NDArray[np.integer]

    def __len__(self) -> int:
        """Return the number of detections."""
        return int(self.conf.shape[0])

    @property
    def xyxy(self) -> NDArray[np.float32]:
        """Return boxes converted from center to corner format."""
        return _xywh_center_to_xyxy(self.xywh)

    def __getitem__(self, item: NumpyIndex) -> UltralyticsResultsLite:
        """Slice detections by boolean mask or index array."""
        return UltralyticsResultsLite(
            conf=np.asarray(self.conf[item]),
            xywh=np.asarray(self.xywh[item]),
            cls=np.asarray(self.cls[item]),
        )


class BotsortTracker(Tracker):
    """Tracker implementation based on ultralytics BoT-SORT.

    Notes:
    - We use ultralytics' BoT-SORT to associate detections to internal track IDs.
    - We keep OT's own lifecycle semantics (`t_min`/`t_miss_max`) so the output
      integrates with the existing buffering/finishing logic.
    - Call :meth:`reset` at each independent video-group boundary;
      GroupedFilesTracker does so before streaming each group.
    """

    def __init__(self, get_current_config: GetCurrentConfig) -> None:
        """Create a BoT-SORT tracker bound to the current application config.

        Args:
            get_current_config (GetCurrentConfig): Accessor for live TrackConfig.
        """
        super().__init__()
        self._get_current_config = get_current_config

        self._botsort: BoTSORTTrackerLike | None = None
        self._frame_rate_by_source: dict[str, int] = {}
        self._class_name_to_id: dict[str, int] = {}
        self._registry = TrackRegistry()

    @property
    def config(self) -> TrackConfig:
        """Return the current track configuration."""
        return self._get_current_config.get().track

    @property
    def lifecycle(self) -> TrackerLifecycle:
        """Return BoT-SORT's own lifecycle thresholds.

        Reads ``TRACK.BOT_SORT`` directly rather than the tracker-dispatching
        ``TrackConfig.lifecycle``, mirroring ``IouTracker``: this adapter is
        unambiguously BoT-SORT.

        Returns:
            TrackerLifecycle: BoT-SORT lifecycle thresholds.
        """
        botsort = self.config.botsort
        return TrackerLifecycle(botsort.t_min, botsort.t_miss_max)

    def reset(self) -> None:
        """Clear all tracker state between independent video groups."""
        self._reset_for_new_group()

    def _reset_for_new_group(self) -> None:
        """Clear Ultralytics state, ID maps, and lifecycle bookkeeping."""
        self._botsort = None
        self._frame_rate_by_source = {}
        self._class_name_to_id = {}
        self._registry.clear()

    def _frame_rate_from_source(self, frame: DetectedFrame) -> int:
        """Read and cache FPS for ``frame.source`` from ``.otdet`` metadata.

        Args:
            frame (DetectedFrame): Frame whose source path carries OTDET metadata.

        Returns:
            int: Rounded positive FPS used to construct BOTSORT.

        Raises:
            ValueError: If the source is not ``.otdet`` or FPS is missing.
        """
        cached = self._frame_rate_by_source.get(frame.source)
        if cached is not None:
            return cached

        source = Path(frame.source)
        if source.suffix.lower() != ".otdet":
            raise ValueError(
                "BoT-SORT requires FPS metadata from an .otdet source file, "
                f"but got '{source.suffix or '<no suffix>'}' "
                f"for source '{frame.source}'."
            )

        try:
            metadata = read_json_bz2_metadata(source)
        except Exception as e:
            raise ValueError(
                "BoT-SORT requires readable .otdet metadata to determine FPS: "
                f"'{frame.source}'."
            ) from e

        extracted = extract_frame_rate_from_metadata(metadata)
        if extracted is None:
            raise ValueError(
                "BoT-SORT requires FPS metadata in .otdet video section "
                f"('{dataformat.ACTUAL_FPS}' or '{dataformat.RECORDED_FPS}') "
                f"for source '{frame.source}'."
            )

        frame_rate = to_frame_rate(extracted)
        self._frame_rate_by_source[frame.source] = frame_rate
        return frame_rate

    def _build_args(self, frame_rate: int) -> types.SimpleNamespace:
        """Build Ultralytics BOTSORT args for ``frame_rate``.

        Args:
            frame_rate (int): Video frame rate.

        Returns:
            types.SimpleNamespace: Namespace expected by ultralytics BOTSORT.
        """
        args_dict = resolve_botsort_tracker_params(self.config.botsort, frame_rate)
        validate_botsort_reid_config(args_dict)
        validate_botsort_gmc_config(args_dict)
        return types.SimpleNamespace(**args_dict)

    def _ensure_botsort_initialized(self, frame: DetectedFrame) -> BoTSORTTrackerLike:
        """Lazily construct the Ultralytics BOTSORT instance for ``frame``.

        Args:
            frame (DetectedFrame): Current frame (used for FPS and optional image).

        Returns:
            BoTSORTTrackerLike: Initialized ultralytics tracker.

        Raises:
            ModuleNotFoundError: If ultralytics is not installed.
            ValueError: If the source FPS cannot be determined, if the
                effective params request unsupported ReID or GMC, or if ReID
                is enabled for a frame that carries no image.
        """
        if self._botsort is not None:
            return self._botsort

        # Lazy import to avoid hard dependency on ultralytics during non-botsort runs.
        try:
            from ultralytics.trackers import BOTSORT  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "ultralytics is required for `--tracker botsort`. Install the optional "
                "dependencies that include ultralytics."
            ) from e

        # Resolve params early so ReID/model validation uses effective values.
        frame_rate = self._frame_rate_from_source(frame)
        args = self._build_args(frame_rate)

        if bool(getattr(args, "with_reid", False)) and frame.image is None:
            raise ValueError(
                "BoT-SORT ReID is enabled in TRACK.BOT_SORT, but the frame "
                "carries no image. .otdet input has no images; provide images "
                "(streaming mode) or disable ReID (`WITH_REID: false`)."
            )

        self._botsort = cast(
            BoTSORTTrackerLike,
            BOTSORT(args=args, frame_rate=frame_rate),
        )
        return self._botsort

    def _class_id(self, label: str) -> int:
        """Map a class label to a stable integer id for Ultralytics.

        Args:
            label (str): Detection class name.

        Returns:
            int: Integer class id local to this tracker instance.
        """
        if label not in self._class_name_to_id:
            self._class_name_to_id[label] = len(self._class_name_to_id)
        return self._class_name_to_id[label]

    def _build_ultralytics_results(
        self, frame: DetectedFrame
    ) -> UltralyticsResultsLite:
        """Convert OTVision detections into an Ultralytics-compatible result object.

        Args:
            frame (DetectedFrame): Frame with untracked detections.

        Returns:
            UltralyticsResultsLite: Result stub with ``conf``/``xywh``/``cls``.
        """
        detections = list(frame.detections)
        n = len(detections)

        xywh: NDArray[np.float32] = np.zeros((n, 4), dtype=np.float32)
        conf: NDArray[np.float32] = np.zeros((n,), dtype=np.float32)
        cls: NDArray[np.int32] = np.zeros((n,), dtype=np.int32)

        for i, det in enumerate(detections):
            # Our detections already store (x, y, w, h) with (x, y)
            # as center coordinates.
            xywh[i] = np.array([det.x, det.y, det.w, det.h], dtype=np.float32)
            conf[i] = float(det.conf)
            cls[i] = self._class_id(det.label)

        # ultralytics BoT-SORT expects `results.conf`, `results.xywh`,
        # and `results.cls`.
        return UltralyticsResultsLite(conf=conf, xywh=xywh, cls=cls)

    @staticmethod
    def _observed_botsort_ids(tracks: NDArray[np.floating] | None) -> set[int]:
        """Collect the Ultralytics track ids present in an ``update()`` result.

        Args:
            tracks (NDArray[np.floating] | None): Validated Nx8 update rows.

        Returns:
            set[int]: Ultralytics track ids observed in the current frame.
        """
        if tracks is None:
            return set()
        return {int(row[_BOTSORT_COL_TRACK_ID]) for row in tracks}

    def _assign_track_ids(
        self,
        tracks: NDArray[np.floating] | None,
        frame_no: FrameNo,
        id_generator: IdGenerator,
    ) -> dict[int, TrackAssignment]:
        """Map each matched detection index to its OTVision track.

        Args:
            tracks (NDArray[np.floating] | None): Validated Nx8 update rows.
            frame_no (FrameNo): Current frame number.
            id_generator (IdGenerator): Provider of new OTVision track ids.

        Returns:
            dict[int, TrackAssignment]: Detection index to assigned track.
        """
        if tracks is None:
            return {}
        return {
            int(row[_BOTSORT_COL_DET_IDX]): self._registry.observe(
                int(row[_BOTSORT_COL_TRACK_ID]), frame_no, id_generator
            )
            for row in tracks
        }

    def track_frame(
        self, frame: DetectedFrame, id_generator: IdGenerator
    ) -> TrackedFrame:
        """Track detections in a single frame with ultralytics BoT-SORT.

        Args:
            frame (DetectedFrame): Frame with untracked detections.
            id_generator (IdGenerator): Provider of new OTVision track ids.

        Returns:
            TrackedFrame: Frame with tracked detections and lifecycle sets.
        """
        botsort = self._ensure_botsort_initialized(frame)
        results = self._build_ultralytics_results(frame)

        # ultralytics expects uint8 images; the detection pipeline guarantees
        # this dtype when images are present.
        tracks = botsort.update(
            results,
            cast(NDArray[np.uint8] | None, frame.image),
        )
        tracks_arr: NDArray[np.floating] | None = (
            None if tracks is None or len(tracks) == 0 else np.asarray(tracks)
        )
        validate_botsort_update_rows(tracks_arr)

        assignments = self._assign_track_ids(tracks_arr, frame.no, id_generator)
        self._registry.age_unobserved(self._observed_botsort_ids(tracks_arr))
        finished_track_ids, discarded_track_ids = self._registry.evict_expired(
            self.lifecycle
        )
        tracked_detections = [
            detection.of_track(assignment.ot_id, is_first=assignment.is_first)
            for index, detection in enumerate(frame.detections)
            if (assignment := assignments.get(index)) is not None
        ]

        return TrackedFrame(
            no=frame.no,
            occurrence=frame.occurrence,
            source=frame.source,
            output=frame.output,
            detections=tracked_detections,
            image=frame.image,
            finished_tracks=finished_track_ids,
            discarded_tracks=discarded_track_ids,
        )
