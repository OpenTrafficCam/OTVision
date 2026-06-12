"""Video-backed tracker decorator for file-mode appearance tracking.

File-based tracking parses ``.otdet`` files into :class:`DetectedFrame`s with
``image=None``; appearance-based trackers (BoT-SORT ReID) require per-frame
images. This decorator lazily reads the sibling video of each ``.otdet``
source one frame per :meth:`track_frame` call, attaches it to the frame for
the wrapped tracker, and strips it from the result so materialized chunks
stay memory-bounded (a 15-min 1280x960 bundle fully decoded would be ~79 GB).

Alignment contract (hard errors, never silent degradation):

- the sibling video must exist (``.mp4``/``.avi``/``.mkv``/``.mov``),
- ``.otdet`` frame numbers per source must be consecutive,
- the video must provide a frame for every detection frame (shorter video
  fails on read; unconsumed trailing video frames fail on source switch).

``.otdet`` frame keys are 1-based and cover every video frame, so sequential
``VideoCapture.read()`` calls align 1:1 with consecutive frame numbers.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import cv2
from numpy import ndarray

from OTVision.domain.frame import DetectedFrame, TrackedFrame
from OTVision.track.model.tracking_interfaces import IdGenerator, Tracker

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")


class VideoSourceError(Exception):
    """Video cannot back the detection stream of a ``.otdet`` source."""


class _SequentialVideoReader:
    """Reads one source's video in lockstep with its detection frames."""

    def __init__(self, otdet_source: str) -> None:
        self._source = otdet_source
        self._video_path = self._resolve_video(Path(otdet_source))
        self._capture = cv2.VideoCapture(str(self._video_path))
        if not self._capture.isOpened():
            raise VideoSourceError(
                f"cannot open video {self._video_path} for {otdet_source}"
            )
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._consumed = 0
        self._last_frame_no: Optional[int] = None

    @staticmethod
    def _resolve_video(otdet: Path) -> Path:
        candidates = [otdet.with_suffix(ext) for ext in VIDEO_EXTENSIONS]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise VideoSourceError(
            f"no sibling video for {otdet} (tried: "
            f"{', '.join(str(c) for c in candidates)})"
        )

    def read(self, frame_no: int) -> ndarray:
        if self._last_frame_no is not None and frame_no != self._last_frame_no + 1:
            raise VideoSourceError(
                f"non-consecutive frame numbers in {self._source}: "
                f"{self._last_frame_no} -> {frame_no} (frame drift?)"
            )
        ok, image = self._capture.read()
        if not ok or image is None:
            raise VideoSourceError(
                f"video {self._video_path} ended at frame {self._consumed} "
                f"but detections continue (frame {frame_no}) — "
                "video shorter than detection stream"
            )
        self._consumed += 1
        self._last_frame_no = frame_no
        return image

    def close(self) -> None:
        remaining = self._frame_count - self._consumed
        self._capture.release()
        # CAP_PROP_FRAME_COUNT is container metadata and may be unreliable;
        # only a positive remainder is treated as a mismatch.
        if remaining > 0:
            raise VideoSourceError(
                f"video {self._video_path} has {remaining} unconsumed frames "
                f"after {self._consumed} detection frames — "
                "video longer than detection stream"
            )


class VideoBackedTracker(Tracker):
    """Decorator attaching video frames to file-based detection streams.

    Wraps any :class:`Tracker`; install when ReID is enabled for file-based
    tracking. ``reid_model`` is validated eagerly: ``"auto"`` relies on
    detector-native features that only exist inside a live YOLO predictor,
    so it can never work in file mode.
    """

    def __init__(self, wrapped: Tracker, reid_model: Optional[str] = None) -> None:
        if reid_model == "auto":
            raise ValueError(
                "BoT-SORT ReID MODEL 'auto' is invalid for file-based "
                "tracking: it requires detector-native features that exist "
                "only in a live YOLO predictor. Configure a real model, "
                "e.g. MODEL: yolo11n-cls.pt"
            )
        self._wrapped = wrapped
        self._reader: Optional[_SequentialVideoReader] = None

    def track_frame(
        self, frame: DetectedFrame, id_generator: IdGenerator
    ) -> TrackedFrame:
        reader = self._reader_for(frame.source)
        image = reader.read(frame.no)
        enriched = replace(frame, image=image)
        result = self._wrapped.track_frame(enriched, id_generator)
        return replace(result, image=None)

    def _reader_for(self, source: str) -> _SequentialVideoReader:
        if self._reader is not None and self._reader._source != source:
            reader, self._reader = self._reader, None
            reader.close()
        if self._reader is None:
            self._reader = _SequentialVideoReader(source)
        return self._reader
