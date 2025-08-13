# ultralytics_trackers_plugin.py

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from OTVision.application.config import TrackConfig
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.domain.detection import Detection, TrackedDetection, TrackId
from OTVision.domain.frame import DetectedFrame, TrackedFrame
from OTVision.track.model.tracking_interfaces import IdGenerator, Tracker

# Ultralytics trackers
try:
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.bot_sort import BOTSORT
except Exception:  # Fallback for older package structures
    from ultralytics.tracker.byte_tracker import BYTETracker  # type: ignore
    from ultralytics.tracker.bot_sort import BOTSORT  # type: ignore


# -------------------------------
# Helpers: geometry and adapters
# -------------------------------


def _detections_to_arrays(
    detections: Sequence[Detection],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Convert OTVision Detection (center x,y,w,h) to arrays Ultralytics understands
    # Returns: xywh (N,4), xyxy (N,4), conf (N,), cls (N,)
    n = len(detections)
    xywh = np.empty((n, 4), dtype=np.float32)
    xyxy = np.empty((n, 4), dtype=np.float32)
    conf = np.empty((n,), dtype=np.float32)
    cls_ids = np.zeros(
        (n,), dtype=np.float32
    )  # if your labels are numeric you can map here

    for i, d in enumerate(detections):
        x, y, w, h = float(d.x), float(d.y), float(d.w), float(d.h)
        xywh[i] = [x, y, w, h]
        xyxy[i] = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        conf[i] = float(d.conf)
        # If label is a str, keep 0.0; if numeric, map it here:
        # cls_ids[i] = float(getattr(d, "cls_id", 0))

    return xywh, xyxy, conf, cls_ids


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a, b: [xmin, ymin, xmax, ymax]
    ixmin = max(a[0], b[0])
    iymin = max(a[1], b[1])
    ixmax = min(a[2], b[2])
    iymax = min(a[3], b[3])
    iw = max(0.0, ixmax - ixmin)
    ih = max(0.0, iymax - iymin)
    inter = iw * ih
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _assign_tracks_to_detections(
    track_xyxy: np.ndarray, det_xyxy: np.ndarray, iou_thresh: float = 0.2
) -> Dict[int, int]:
    """
    Greedy assign each track to at most one detection by IoU.
    Returns: map track_index -> detection_index
    """
    if track_xyxy.size == 0 or det_xyxy.size == 0:
        return {}

    t = track_xyxy.shape[0]
    d = det_xyxy.shape[0]
    iou_mat = np.zeros((t, d), dtype=np.float32)
    for i in range(t):
        for j in range(d):
            iou_mat[i, j] = _iou_xyxy(track_xyxy[i], det_xyxy[j])

    assigned: Dict[int, int] = {}
    used_dets = set()
    # Greedy by best IoU per track
    # Sort track indices by their max IoU descending to prefer strong matches first
    order = np.argsort(-np.max(iou_mat, axis=1))
    for ti in order:
        dj = int(np.argmax(iou_mat[ti]))
        if dj not in used_dets and iou_mat[ti, dj] >= iou_thresh:
            assigned[ti] = dj
            used_dets.add(dj)
    return assigned


class _ULResultsAdapter:
    """
    Minimal adapter to mimic the YOLO Results/Boxes API used by Ultralytics trackers.
    Supports:
      - len()
      - [boolean_mask] indexing
      - attributes: xywh (N,4), xyxy (N,4), conf (N,), cls (N,)
    """

    def __init__(
        self,
        xywh: np.ndarray,
        conf: np.ndarray,
        cls_ids: np.ndarray,
        xyxy: Optional[np.ndarray] = None,
    ):
        self.xywh = xywh.astype(np.float32, copy=False)
        self.conf = conf.astype(np.float32, copy=False)
        self.cls = cls_ids.astype(np.float32, copy=False)
        self.xyxy = (
            xyxy.astype(np.float32, copy=False)
            if xyxy is not None
            else self._xywh_to_xyxy(self.xywh)
        )

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx):
        # idx can be boolean mask or integer indices
        return _ULResultsAdapter(
            xywh=self.xywh[idx],
            conf=self.conf[idx],
            cls_ids=self.cls[idx],
            xyxy=self.xyxy[idx],
        )

    @staticmethod
    def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        xyxy = np.empty_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
        return xyxy


# --------------------------------
# Base adapter for Ultralytics
# --------------------------------


class _BaseUltralyticsTrackerAdapter(Tracker):
    """
    Base class that adapts an Ultralytics tracker to OTVision's Tracker interface.
    Subclasses must implement _create_ul_tracker(self) -> object.
    """

    def __init__(self, get_current_config: GetCurrentConfig):
        super().__init__()
        self._get_current_config = get_current_config

        # Underlying Ultralytics tracker instance
        self._ul_tracker = self._create_ul_tracker()

        # Map Ultralytics track_id -> OTVision TrackId
        self._ul_to_ot: Dict[int, TrackId] = {}

        # Keep sets to figure out finished tracks (those that disappear from active set)
        self._prev_active_ul_ids: set[int] = set()

    @property
    def config(self) -> TrackConfig:
        return self._get_current_config.get().track

    def _create_ul_tracker(self):
        raise NotImplementedError

    def _ensure_ot_id(self, ul_id: int, id_generator: IdGenerator) -> TrackId:
        if ul_id not in self._ul_to_ot:
            self._ul_to_ot[ul_id] = next(id_generator)
        return self._ul_to_ot[ul_id]

    def _current_active_ul_ids(self) -> set[int]:
        # Active tracks at end of update; Ultralytics keeps them in tracked_stracks
        # Filter for those is_activated True to mirror returned outputs
        active_ids: set[int] = set()
        try:
            for st in getattr(self._ul_tracker, "tracked_stracks", []):
                if getattr(st, "is_activated", False):
                    active_ids.add(int(st.track_id))
        except Exception:
            # Fallback: parse from returned results if needed
            pass
        return active_ids

    def _ul_tracks_to_arrays(
        self, ul_tracks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        ul_tracks rows are [xyxy(4), track_id, score, cls, idx] per STrack.result
        Returns: xyxy (N,4), ul_ids (N,), scores (N,), cls (N,)
        """
        if ul_tracks.size == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        xyxy = ul_tracks[:, 0:4].astype(np.float32, copy=False)
        ul_ids = ul_tracks[:, 4].astype(np.int64, copy=False)
        scores = ul_tracks[:, 5].astype(np.float32, copy=False)
        cls_ids = ul_tracks[:, 6].astype(np.float32, copy=False)
        return xyxy, ul_ids, scores, cls_ids

    def _make_tracked_detections(
        self,
        frame: DetectedFrame,
        ul_tracks_result: np.ndarray,
        id_generator: IdGenerator,
    ) -> List[TrackedDetection]:
        # Convert returned UL tracks into OTVision TrackedDetection by matching them
        # back to input detections via IoU.
        tracked: List[TrackedDetection] = []
        if len(frame.detections) == 0:
            return tracked

        track_xyxy, ul_ids, _, _ = self._ul_tracks_to_arrays(ul_tracks_result)
        if track_xyxy.shape[0] == 0:
            return tracked

        # Prepare detection boxes
        _, det_xyxy, _, _ = _detections_to_arrays(frame.detections)

        # Greedy IoU matching: track index -> detection index
        assignment = _assign_tracks_to_detections(track_xyxy, det_xyxy, iou_thresh=0.2)
        # For each matched track, produce TrackedDetection
        for ti, di in assignment.items():
            ul_id = int(ul_ids[ti])
            ot_id = self._ensure_ot_id(ul_id, id_generator)
            det = frame.detections[di]
            is_new = ul_id not in self._prev_active_ul_ids
            tracked.append(det.of_track(ot_id, is_new))

        return tracked

    def track_frame(
        self, frame: DetectedFrame, id_generator: IdGenerator
    ) -> TrackedFrame:
        # Build Ultralytics-style results adapter
        xywh, xyxy, conf, cls_ids = _detections_to_arrays(frame.detections)
        ul_results = _ULResultsAdapter(xywh=xywh, conf=conf, cls_ids=cls_ids, xyxy=xyxy)

        # Call underlying Ultralytics tracker
        # For BYTETracker, img is optional; BoT-SORT will use it for GMC if provided
        try:
            ul_tracks_result = self._ul_tracker.update(
                ul_results, getattr(frame, "image", None), None
            )
        except TypeError:
            # Older signatures may be update(results) only
            ul_tracks_result = self._ul_tracker.update(ul_results)

        # Build TrackedDetection list by mapping tracks to detections
        tracked_detections = self._make_tracked_detections(
            frame, ul_tracks_result, id_generator
        )

        # Compute finished tracks (those active previously but not now)
        current_active = self._current_active_ul_ids()
        finished_ul = self._prev_active_ul_ids - current_active
        finished_tracks: set[TrackId] = {
            self._ul_to_ot[ul] for ul in finished_ul if ul in self._ul_to_ot
        }
        self._prev_active_ul_ids = current_active

        # Discarded tracks: not tracked here; leave empty
        discarded_tracks: set[TrackId] = set()

        return TrackedFrame(
            no=frame.no,
            occurrence=frame.occurrence,
            source=frame.source,
            output=frame.output,
            detections=tracked_detections,
            image=frame.image,
            finished_tracks=finished_tracks,
            discarded_tracks=discarded_tracks,
        )


# --------------------------------
# BYTETracker plugin
# --------------------------------


class ByteTrackTracker(_BaseUltralyticsTrackerAdapter):
    """
    OTVision plugin that wraps Ultralytics BYTETracker.

    Default thresholds mirror common ByteTrack configs; tweak as needed
    via the _create_ul_tracker args block.
    """

    def _create_ul_tracker(self):
        # Build args SimpleNamespace as expected by Ultralytics BYTETracker
        # You can wire these to your OTVision config if available.
        args = SimpleNamespace(
            # tracking thresholds
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            match_thresh=0.8,
            # buffer
            track_buffer=30,
            # misc
            fuse_score=False,
        )
        return BYTETracker(args=args, frame_rate=30)


# --------------------------------
# BoT-SORT tracker plugin
# --------------------------------


class BoTSORTTracker(_BaseUltralyticsTrackerAdapter):
    """
    OTVision plugin that wraps Ultralytics BOTSORT (with GMC).
    By default, ReID is disabled to keep dependencies minimal.

    If you want ReID, set with_reid=True and configure model in args block.
    """

    def _create_ul_tracker(self):
        args = SimpleNamespace(
            # ByteTrack-like thresholds
            track_high_thresh=0.2,
            track_low_thresh=0.05,
            new_track_thresh=0.25,
            track_buffer=120,
            match_thresh=0.99,
            fuse_score=True,
            # BoT-SORT specifics
            proximity_thresh=0.1,  # IoU gating
            appearance_thresh=0.25,  # ReID gating
            with_reid=True,  # default off
            model="auto",  # used only if with_reid=True
            gmc_method="sparseOptFlow",  # 'sparseOptFlow' | 'orb' | 'sift' | 'ecc'
        )
        return BOTSORT(args=args, frame_rate=20)
