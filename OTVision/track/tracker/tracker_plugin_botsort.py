from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import types
from typing import Any, Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from OTVision import dataformat
from OTVision.application.config import TrackConfig
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.domain.detection import TrackId, TrackedDetection
from OTVision.domain.frame import DetectedFrame, TrackedFrame
from OTVision.helpers.files import read_json_bz2_metadata
from OTVision.track.model.tracking_interfaces import IdGenerator, Tracker


UltralyticsScalar = bool | int | float | str


def _xywh_center_to_xyxy(xywh: NDArray[np.floating]) -> NDArray[np.float32]:
    """Map center-format ``(x, y, w, h)`` boxes to ``(x1, y1, x2, y2)``.

    Matches :func:`ultralytics.utils.ops.xywh2xyxy` (used by BYTETracker / BoT-SORT).
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


def _extract_frame_rate_from_metadata(metadata: dict) -> float | None:
    """Extract preferred frame rate from OTDET metadata."""
    video = metadata.get(dataformat.VIDEO, {})
    if not isinstance(video, dict):
        return None

    actual = video.get(dataformat.ACTUAL_FPS)
    if isinstance(actual, (int, float)) and float(actual) > 0:
        return float(actual)

    recorded = video.get(dataformat.RECORDED_FPS)
    if isinstance(recorded, (int, float)) and float(recorded) > 0:
        return float(recorded)

    return None


class UltralyticsResultsLike(Protocol):
    """Subset of ultralytics ``Results`` API required by ``BYTETracker.update``."""

    conf: NDArray[np.floating]
    xywh: NDArray[np.floating]
    cls: NDArray[np.integer]

    @property
    def xyxy(self) -> NDArray[np.floating]:
        """Needed when ``img is not None`` (GMC uses ``results.xyxy``)."""
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, item: Any) -> UltralyticsResultsLike:
        """Boolean mask indexing as in ``results = results[remain_inds]``."""
        ...


class BoTSORTTrackerLike(Protocol):
    # ultralytics BYTETracker/BOTSORT (8.3+): update(results, img=None)
    def update(
        self,
        results: UltralyticsResultsLike,
        img: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.floating]:
        ...


class TrackLifecycleState(TypedDict):
    first_frame: int
    last_frame: int
    age_missing: int
    max_conf: float


@dataclass
class UltralyticsResultsLite:
    """Minimal stand-in for ultralytics detection results (``conf``/``xywh``/``cls`` + slicing)."""

    conf: NDArray[np.floating]
    xywh: NDArray[np.floating]
    cls: NDArray[np.integer]

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    @property
    def xyxy(self) -> NDArray[np.float32]:
        return _xywh_center_to_xyxy(self.xywh)

    def __getitem__(self, item: Any) -> UltralyticsResultsLite:
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
    """

    # Defaults mirrored from `ultralytics/cfg/trackers/botsort.yaml`.
    # These ensure we can build a working tracker even if the YAML only specifies
    # a subset of fields.
    _DEFAULT_BOTSORT_ARGS: dict[str, UltralyticsScalar] = {
        "tracker_type": "botsort",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        # `track_buffer` is aligned to `t_miss_max` by default (see _build_args()).
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.8,
        "with_reid": False,
        "model": "auto",
    }

    def __init__(self, get_current_config: GetCurrentConfig) -> None:
        super().__init__()
        self._get_current_config = get_current_config

        self._botsort: BoTSORTTrackerLike | None = None
        self._frame_rate_by_source: dict[str, int] = {}
        self._class_name_to_id: dict[str, int] = {}
        self._botsort_track_id_to_ot_id: dict[int, TrackId] = {}
        self._ot_id_to_botsort_track_id: dict[TrackId, int] = {}

        # Lifecycle state keyed by OT track id.
        # Values are: {first_frame, last_frame, age_missing, max_conf}
        self._track_state: dict[TrackId, TrackLifecycleState] = {}

    @property
    def config(self) -> TrackConfig:
        return self._get_current_config.get().track

    def _reset_for_new_group(self) -> None:
        self._botsort = None
        self._frame_rate_by_source = {}
        self._class_name_to_id = {}
        self._botsort_track_id_to_ot_id = {}
        self._ot_id_to_botsort_track_id = {}
        self._track_state = {}

    def _frame_rate_from_source(self, frame: DetectedFrame) -> int:
        cached = self._frame_rate_by_source.get(frame.source)
        if cached is not None:
            return cached

        source = Path(frame.source)
        if source.suffix.lower() != ".otdet":
            raise ValueError(
                "BoT-SORT requires FPS metadata from an .otdet source file, "
                f"but got '{source.suffix or '<no suffix>'}' for source '{frame.source}'."
            )

        try:
            metadata = read_json_bz2_metadata(source)
        except Exception as e:
            raise ValueError(
                f"BoT-SORT requires readable .otdet metadata to determine FPS: '{frame.source}'."
            ) from e

        extracted = _extract_frame_rate_from_metadata(metadata)
        if extracted is None:
            raise ValueError(
                "BoT-SORT requires FPS metadata in .otdet video section "
                f"('{dataformat.ACTUAL_FPS}' or '{dataformat.RECORDED_FPS}') for source '{frame.source}'."
            )

        frame_rate = max(1, int(round(extracted)))
        self._frame_rate_by_source[frame.source] = frame_rate
        return frame_rate

    def _build_args(self) -> types.SimpleNamespace:
        args_dict = dict(self._DEFAULT_BOTSORT_ARGS)
        tracker_params = cast(
            dict[str, UltralyticsScalar], self.config.botsort.tracker_params
        )
        args_dict.update(tracker_params)

        # Align ultralytics track-buffer with OT's "missing frames" semantics by default.
        if "track_buffer" not in self.config.botsort.tracker_params:
            args_dict["track_buffer"] = int(self.config.botsort.t_miss_max)

        # ultralytics expects a Namespace-like object.
        return types.SimpleNamespace(**args_dict)

    def _ensure_botsort_initialized(self, frame: DetectedFrame) -> BoTSORTTrackerLike:
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

        if self.config.botsort.tracker_params.get("with_reid", False) and frame.image is None:
            raise ValueError(
                "BoT-SORT ReID is enabled in TRACK.BOT_SORT, but frame.image is missing. "
                "Provide images (streaming mode) or disable ReID (`with_reid: false`)."
            )

        frame_rate = self._frame_rate_from_source(frame)
        self._botsort = cast(
            BoTSORTTrackerLike, BOTSORT(args=self._build_args(), frame_rate=frame_rate)
        )
        return self._botsort

    def _class_id(self, label: str) -> int:
        if label not in self._class_name_to_id:
            self._class_name_to_id[label] = len(self._class_name_to_id)
        return self._class_name_to_id[label]

    def _build_ultralytics_results(
        self, frame: DetectedFrame
    ) -> UltralyticsResultsLite:
        detections = list(frame.detections)
        n = len(detections)

        xywh: NDArray[np.float32] = np.zeros((n, 4), dtype=np.float32)
        conf: NDArray[np.float32] = np.zeros((n,), dtype=np.float32)
        cls: NDArray[np.int32] = np.zeros((n,), dtype=np.int32)

        for i, det in enumerate(detections):
            # Our detections already store (x, y, w, h) with (x,y) as center coordinates.
            xywh[i] = np.array([det.x, det.y, det.w, det.h], dtype=np.float32)
            conf[i] = float(det.conf)
            cls[i] = self._class_id(det.label)

        # ultralytics BoT-SORT expects `results.conf`, `results.xywh`, and `results.cls`.
        return UltralyticsResultsLite(conf=conf, xywh=xywh, cls=cls)

    def track_frame(
        self, frame: DetectedFrame, id_generator: IdGenerator
    ) -> TrackedFrame:
        if frame.no == 0:
            self._reset_for_new_group()

        botsort = self._ensure_botsort_initialized(frame)
        results = self._build_ultralytics_results(frame)

        # `BOTSORT.update()` returns Nx8 float rows:
        # [x1, y1, x2, y2, track_id, score, cls, det_idx]
        # The concrete dtype/shape is ultralytics-dependent, but we only index by
        # known column offsets.
        tracks = botsort.update(
            results,
            cast(NDArray[np.uint8] | None, frame.image),
        )

        det_idx_to_ot_id: dict[int, TrackId] = {}

        if tracks is not None and len(tracks) > 0:
            for row in tracks:
                det_idx = int(row[-1])
                botsort_track_id = int(row[4])

                if botsort_track_id not in self._botsort_track_id_to_ot_id:
                    ot_id = next(id_generator)
                    self._botsort_track_id_to_ot_id[botsort_track_id] = ot_id
                    self._ot_id_to_botsort_track_id[ot_id] = botsort_track_id
                else:
                    ot_id = self._botsort_track_id_to_ot_id[botsort_track_id]

                det_idx_to_ot_id[det_idx] = ot_id

                # Update/update-on-hit lifecycle state immediately.
                state = self._track_state.get(ot_id)
                score = float(row[5])
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
            # (IOU tracker saves on `age_missing == t_miss_max`.)
            if state["age_missing"] <= self.config.t_miss_max:
                continue

            span = int(state["last_frame"]) - int(state["first_frame"])
            if span >= self.config.t_min:
                finished_track_ids.add(ot_id)
            else:
                discarded_track_ids.add(ot_id)

            botsort_track_id = self._ot_id_to_botsort_track_id.get(ot_id)
            if botsort_track_id is not None:
                self._botsort_track_id_to_ot_id.pop(botsort_track_id, None)
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
                    is_first=int(self._track_state[ot_id]["first_frame"]) == frame.no
                    if ot_id in self._track_state
                    else True,
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

