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
from math import ceil
from pathlib import Path
from typing import Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from OTVision import dataformat
from OTVision.application.config import (
    DEFAULT_BOTSORT_TRACKER_PARAMS,
    BotSortTrackerParam,
    TrackConfig,
    _TrackBotSortConfig,
)
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.domain.detection import TrackedDetection, TrackId
from OTVision.domain.frame import DetectedFrame, TrackedFrame
from OTVision.helpers.files import read_json_bz2_metadata
from OTVision.helpers.log import LOGGER_NAME
from OTVision.track.model.tracking_interfaces import IdGenerator, Tracker

log = logging.getLogger(LOGGER_NAME)

UltralyticsScalar = BotSortTrackerParam
NumpyIndex = int | slice | NDArray[np.bool_] | NDArray[np.integer]

# Ultralytics BOTSORT.update() (8.3.159) returns Nx8 float rows:
# [x1, y1, x2, y2, track_id, score, cls, det_idx]
_BOTSORT_UPDATE_COLS = 8
_BOTSORT_COL_TRACK_ID = 4
_BOTSORT_COL_SCORE = 5
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


def _try_positive_float(value: object) -> float | None:
    """Return *value* as a positive float, or ``None`` if conversion fails.

    Args:
        value (object): Candidate numeric value (int, float, Decimal, or
            numeric string).

    Returns:
        float | None: Positive float, or ``None`` when conversion fails.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        # ``.otdet`` metadata may store FPS as Decimal via ijson.
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def extract_frame_rate_from_metadata(metadata: dict) -> float | None:
    """Extract preferred frame rate from OTDET metadata.

    Prefers ``actual_fps`` over ``recorded_fps`` when both are present and positive.

    Args:
        metadata (dict): OTDET metadata dictionary.

    Returns:
        float | None: Positive FPS value, or ``None`` if unavailable.
    """
    video = metadata.get(dataformat.VIDEO, {})
    if not isinstance(video, dict):
        return None

    actual = _try_positive_float(video.get(dataformat.ACTUAL_FPS))
    if actual is not None:
        return actual

    recorded = _try_positive_float(video.get(dataformat.RECORDED_FPS))
    if recorded is not None:
        return recorded

    return None


# Keep private alias for older test imports during transition.
_extract_frame_rate_from_metadata = extract_frame_rate_from_metadata


def ultralytics_effective_miss_frames(frame_rate: int, track_buffer: int) -> int:
    """Compute Ultralytics' occlusion window for a ``track_buffer``.

    Ultralytics uses ``int(fps / 30 * track_buffer)``.

    Args:
        frame_rate (int): Video frame rate used by BOTSORT.
        track_buffer (int): Ultralytics ``track_buffer`` argument.

    Returns:
        int: Effective number of missing frames before Ultralytics drops a track.
    """
    fps = max(1, int(frame_rate))
    return int(fps / 30 * int(track_buffer))


def derive_track_buffer(t_miss_max: int, frame_rate: int) -> int:
    """Derive Ultralytics ``track_buffer`` covering at least ``t_miss_max``.

    Ultralytics floors ``int(fps / 30 * track_buffer)``, so ``round`` is not an inverse.
    Using ``ceil(t_miss_max * 30 / fps)`` guarantees the effective lifetime is at least
    ``t_miss_max``. Exact equality is impossible for some FPS values.

    Args:
        t_miss_max (int): OTVision missing-frame threshold.
        frame_rate (int): Video frame rate used by BOTSORT.

    Returns:
        int: Derived ``track_buffer`` (>= 1).
    """
    fps = max(1, int(frame_rate))
    buffer = max(1, int(ceil(t_miss_max * 30 / fps)))
    # Guard floating-point edge cases so effective lifetime never undershoots.
    while ultralytics_effective_miss_frames(fps, buffer) < t_miss_max:
        buffer += 1
    return buffer


def resolve_botsort_tracker_params(
    botsort_config: _TrackBotSortConfig,
    frame_rate: int,
) -> dict[str, UltralyticsScalar]:
    """Resolve effective BoT-SORT args (defaults + overrides + derived buffer).

    This is the single source of truth for tracker construction and ``.ottrk``
    metadata. When YAML does not set ``track_buffer``, it is derived from
    ``t_miss_max`` and ``frame_rate``.

    Args:
        botsort_config (_TrackBotSortConfig): BoT-SORT configuration section.
        frame_rate (int): Video frame rate used by BOTSORT.

    Returns:
        dict[str, UltralyticsScalar]: Effective Ultralytics tracker arguments.
    """
    args_dict: dict[str, UltralyticsScalar] = dict(DEFAULT_BOTSORT_TRACKER_PARAMS)
    tracker_params = cast(dict[str, UltralyticsScalar], botsort_config.tracker_params)
    args_dict.update(tracker_params)

    derived_buffer = derive_track_buffer(botsort_config.t_miss_max, frame_rate)
    if "track_buffer" not in botsort_config.tracker_params:
        args_dict["track_buffer"] = derived_buffer
    else:
        explicit_buffer = int(args_dict["track_buffer"])
        effective_miss = ultralytics_effective_miss_frames(frame_rate, explicit_buffer)
        if effective_miss != botsort_config.t_miss_max:
            log.warning(
                "TRACK.BOT_SORT.TRACK_BUFFER=%s at %s fps implies an Ultralytics "
                "occlusion window of %s frames, but T_MISS_MAX=%s. "
                "Divergent lifecycles can split one physical object into multiple "
                "OTVision track IDs. Omit TRACK_BUFFER to derive %s automatically.",
                explicit_buffer,
                frame_rate,
                effective_miss,
                botsort_config.t_miss_max,
                derived_buffer,
            )

    return args_dict


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


def validate_botsort_reid_config(
    tracker_params: dict[str, UltralyticsScalar],
) -> None:
    """Reject ReID configurations unsupported by OTVision's detection path.

    Ultralytics 8.3.159 interprets ``model: auto`` as native detector feature
    tensors. OTVision only supplies NumPy frame images, so ``auto`` crashes.

    Args:
        tracker_params (dict[str, UltralyticsScalar]): Effective or configured
            tracker params.

    Raises:
        ValueError: If ReID is enabled with ``model: auto``.
    """
    with_reid = bool(tracker_params.get("with_reid", False))
    model = str(tracker_params.get("model", "auto")).lower()
    if with_reid and model == "auto":
        raise ValueError(
            "BoT-SORT ReID with MODEL=auto is not supported: Ultralytics 8.3.159 "
            "expects native detector feature tensors, but OTVision supplies NumPy "
            "frame images. Set an explicit ReID model path/name, or disable ReID "
            "(`WITH_REID: false`)."
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


class TrackLifecycleState(TypedDict):
    """Per-track lifecycle bookkeeping used by :class:`BotsortTracker`."""

    first_frame: int
    last_frame: int
    age_missing: int
    max_conf: float


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
    - Call :meth:`reset` at each independent video-group boundary. As a defensive
      fallback, ``frame.no == 0`` also resets state (GroupedFilesTracker starts
      each group at frame 0).
    """

    def __init__(self, get_current_config: GetCurrentConfig) -> None:
        """Create a BoT-SORT tracker bound to the current application config.

        Args:
            get_current_config (GetCurrentConfig): Accessor for live TrackConfig.
        """
        super().__init__()
        self._get_current_config = get_current_config

        self._botsort: BoTSORTTrackerLike | None = None
        self._resolved_tracker_params: dict[str, UltralyticsScalar] | None = None
        self._frame_rate_by_source: dict[str, int] = {}
        self._class_name_to_id: dict[str, int] = {}
        self._botsort_track_id_to_ot_id: dict[int, TrackId] = {}
        self._ot_id_to_botsort_track_id: dict[TrackId, int] = {}

        # Lifecycle state keyed by OT track id.
        self._track_state: dict[TrackId, TrackLifecycleState] = {}

    @property
    def config(self) -> TrackConfig:
        """Return the current track configuration."""
        return self._get_current_config.get().track

    @property
    def resolved_tracker_params(self) -> dict[str, UltralyticsScalar] | None:
        """Effective Ultralytics args used after the tracker was initialized."""
        return self._resolved_tracker_params

    def reset(self) -> None:
        """Clear all tracker state between independent video groups."""
        self._reset_for_new_group()

    def _reset_for_new_group(self) -> None:
        """Clear Ultralytics state, ID maps, and lifecycle bookkeeping."""
        self._botsort = None
        self._resolved_tracker_params = None
        self._frame_rate_by_source = {}
        self._class_name_to_id = {}
        self._botsort_track_id_to_ot_id = {}
        self._ot_id_to_botsort_track_id = {}
        self._track_state = {}

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

        frame_rate = max(1, int(round(extracted)))
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
        self._resolved_tracker_params = dict(args_dict)
        return types.SimpleNamespace(**args_dict)

    def _ensure_botsort_initialized(self, frame: DetectedFrame) -> BoTSORTTrackerLike:
        """Lazily construct the Ultralytics BOTSORT instance for ``frame``.

        Args:
            frame (DetectedFrame): Current frame (used for FPS and optional image).

        Returns:
            BoTSORTTrackerLike: Initialized ultralytics tracker.

        Raises:
            ModuleNotFoundError: If ultralytics is not installed.
            ValueError: If ReID is enabled without a frame image.
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
                "BoT-SORT ReID is enabled in TRACK.BOT_SORT, but "
                "frame.image is missing. "
                "Provide images (streaming mode) or disable ReID "
                "(`with_reid: false`)."
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
        # Defensive fallback: GroupedFilesTracker starts each video group at
        # frame.no == 0. Prefer calling :meth:`reset` explicitly at group
        # boundaries; this keeps streaming / alternate callers safe.
        if frame.no == 0:
            self._reset_for_new_group()

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

        det_idx_to_ot_id: dict[int, TrackId] = {}

        if tracks_arr is not None:
            for row in tracks_arr:
                det_idx = int(row[_BOTSORT_COL_DET_IDX])
                botsort_track_id = int(row[_BOTSORT_COL_TRACK_ID])

                if botsort_track_id not in self._botsort_track_id_to_ot_id:
                    ot_id = next(id_generator)
                    self._botsort_track_id_to_ot_id[botsort_track_id] = ot_id
                    self._ot_id_to_botsort_track_id[ot_id] = botsort_track_id
                else:
                    ot_id = self._botsort_track_id_to_ot_id[botsort_track_id]

                det_idx_to_ot_id[det_idx] = ot_id

                # Update/update-on-hit lifecycle state immediately.
                state = self._track_state.get(ot_id)
                score = float(row[_BOTSORT_COL_SCORE])
                if state is None:
                    self._track_state[ot_id] = cast(
                        TrackLifecycleState,
                        {
                            "first_frame": frame.no,
                            "last_frame": frame.no,
                            "age_missing": 0,
                            "max_conf": score,
                        },
                    )
                else:
                    state["last_frame"] = frame.no
                    state["age_missing"] = 0
                    state["max_conf"] = max(float(state["max_conf"]), score)

        seen_now: set[TrackId] = set(det_idx_to_ot_id.values())

        # Age tracks that were not observed in the current frame.
        for ot_id, state in list(self._track_state.items()):
            if ot_id in seen_now:
                continue
            state["age_missing"] += 1

        finished_track_ids: set[TrackId] = set()
        discarded_track_ids: set[TrackId] = set()

        # Move tracks that exceeded missing-frame threshold.
        for ot_id, state in list(self._track_state.items()):
            # Match IOU-tracker semantics: we only finish/discard once the track
            # has been missing for *more* than `t_miss_max` consecutive frames.
            # (IOU tracker keeps tracks while `track_age < t_miss_max`, incrementing
            # after the check, so a track with age == t_miss_max is finished on the
            # next miss.  Here we increment first, so `<= t_miss_max` is equivalent.)
            if state["age_missing"] <= self.config.t_miss_max:
                continue

            # NOTE: Unlike the IOU tracker, we intentionally omit a `sigma_h`
            # (high-confidence) gate here.  The IOU tracker associates *all*
            # detections above `sigma_l` and retroactively discards tracks whose
            # `max_conf` never reached `sigma_h`.  BoT-SORT, however, already
            # filters detections during association via `track_high_thresh`,
            # `track_low_thresh`, and `new_track_thresh` — only sufficiently
            # confident detections survive long enough to form tracks, making an
            # additional confidence gate at finish time redundant.
            span = int(state["last_frame"]) - int(state["first_frame"])
            if span >= self.config.t_min:
                finished_track_ids.add(ot_id)
            else:
                discarded_track_ids.add(ot_id)

            mapped_botsort_track_id = self._ot_id_to_botsort_track_id.get(ot_id)
            if mapped_botsort_track_id is not None:
                self._botsort_track_id_to_ot_id.pop(mapped_botsort_track_id, None)
            self._ot_id_to_botsort_track_id.pop(ot_id, None)
            self._track_state.pop(ot_id, None)

        tracked_detections: list[TrackedDetection] = []
        for det_idx, det in enumerate(frame.detections):
            if det_idx not in det_idx_to_ot_id:
                continue
            ot_id = det_idx_to_ot_id[det_idx]

            # "is_first" is true for the first time we ever assigned an OT id.
            tracked_detections.append(
                det.of_track(
                    ot_id,
                    is_first=(
                        int(self._track_state[ot_id]["first_frame"]) == frame.no
                        if ot_id in self._track_state
                        else True
                    ),
                )
            )

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
