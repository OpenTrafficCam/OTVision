"""Unit and integration tests for the video-backed tracker decorator.

File-mode tracking parses ``.otdet`` files into DetectedFrames with
``image=None``; BoT-SORT ReID needs per-frame images. ``VideoBackedTracker``
wraps a tracker and lazily attaches frames read sequentially from the sibling
video of each ``.otdet`` source, stripping the image from the result so chunks
stay memory-bounded. Misalignment (missing video, video shorter than the
detection stream, non-consecutive frame numbers, undersized video on source
switch) is a hard error by design — silently degraded appearance features
would poison A/B comparisons downstream.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from OTVision.domain.detection import Detection  # noqa: E402
from OTVision.domain.frame import DetectedFrame, TrackedFrame  # noqa: E402
from OTVision.track.model.tracking_interfaces import IdGenerator, Tracker  # noqa: E402
from OTVision.track.tracker.video_backed_tracker import (  # noqa: E402
    VideoBackedTracker,
    VideoSourceError,
)
from tests.track.helper.data_builder import create_frame  # noqa: E402

FRAME_W = 64
FRAME_H = 48


def write_video(path: Path, n_frames: int) -> None:
    """Write n solid-color frames; frame i has pixel value i * 40
    (gap >> codec quantization error, so frames stay distinguishable)."""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (FRAME_W, FRAME_H),
    )
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(np.full((FRAME_H, FRAME_W, 3), i * 40, dtype=np.uint8))
    writer.release()


def detection() -> Detection:
    return Detection(label="car", conf=0.9, x=32.0, y=24.0, w=16.0, h=12.0)


def frame_for(no: int, otdet: Path) -> DetectedFrame:
    return create_frame(no, [detection()], input_file_path=otdet)


class SpyTracker(Tracker):
    """Records the images it was given; passes the image through unchanged
    so tests can verify the wrapper strips it from results."""

    def __init__(self) -> None:
        self.seen_images: list = []

    def track_frame(
        self, frame: DetectedFrame, id_generator: IdGenerator
    ) -> TrackedFrame:
        self.seen_images.append(frame.image)
        return TrackedFrame(
            no=frame.no,
            occurrence=frame.occurrence,
            source=frame.source,
            output=frame.output,
            detections=[],
            finished_tracks=set(),
            discarded_tracks=set(),
            image=frame.image,
        )


def ids() -> IdGenerator:
    return iter(range(1, 10_000))


@pytest.fixture
def otdet_with_video(tmp_path: Path) -> Path:
    otdet = tmp_path / "clip.otdet"
    otdet.touch()
    write_video(otdet.with_suffix(".mp4"), 5)
    return otdet


class TestVideoBackedTracker:
    def test_attaches_video_frames_in_order_and_strips_result(
        self, otdet_with_video: Path
    ) -> None:
        spy = SpyTracker()
        tracker = VideoBackedTracker(spy)
        id_gen = ids()

        results = [
            tracker.track_frame(frame_for(no, otdet_with_video), id_gen)
            for no in range(1, 6)
        ]

        assert len(spy.seen_images) == 5
        for i, image in enumerate(spy.seen_images):
            assert image is not None, f"frame {i + 1} got no image"
            assert image.shape == (FRAME_H, FRAME_W, 3)
            assert abs(int(image[0, 0, 0]) - i * 40) <= 20  # codec tolerance
        assert all(r.image is None for r in results)

    def test_missing_video_raises_with_tried_path(self, tmp_path: Path) -> None:
        otdet = tmp_path / "clip.otdet"
        otdet.touch()
        tracker = VideoBackedTracker(SpyTracker())

        with pytest.raises(VideoSourceError, match="clip"):
            tracker.track_frame(frame_for(1, otdet), ids())

    def test_video_shorter_than_stream_raises(self, tmp_path: Path) -> None:
        otdet = tmp_path / "clip.otdet"
        otdet.touch()
        write_video(otdet.with_suffix(".mp4"), 3)
        tracker = VideoBackedTracker(SpyTracker())
        id_gen = ids()

        for no in range(1, 4):
            tracker.track_frame(frame_for(no, otdet), id_gen)
        with pytest.raises(VideoSourceError, match="frame"):
            tracker.track_frame(frame_for(4, otdet), id_gen)

    def test_non_consecutive_frame_numbers_raise(self, otdet_with_video: Path) -> None:
        tracker = VideoBackedTracker(SpyTracker())
        id_gen = ids()

        tracker.track_frame(frame_for(1, otdet_with_video), id_gen)
        with pytest.raises(VideoSourceError, match="consecutive"):
            tracker.track_frame(frame_for(3, otdet_with_video), id_gen)

    def test_unconsumed_video_frames_raise_on_source_switch(
        self, otdet_with_video: Path, tmp_path: Path
    ) -> None:
        next_otdet = tmp_path / "next.otdet"
        next_otdet.touch()
        write_video(next_otdet.with_suffix(".mp4"), 2)
        tracker = VideoBackedTracker(SpyTracker())
        id_gen = ids()

        for no in range(1, 4):  # consume 3 of 5 video frames
            tracker.track_frame(frame_for(no, otdet_with_video), id_gen)
        with pytest.raises(VideoSourceError, match="unconsumed"):
            tracker.track_frame(frame_for(1, next_otdet), id_gen)

    def test_reid_model_auto_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="auto"):
            VideoBackedTracker(SpyTracker(), reid_model="auto")


ultralytics_available = importlib.util.find_spec("ultralytics") is not None


@pytest.mark.skipif(not ultralytics_available, reason="ultralytics required")
class TestVideoBackedBotsortIntegration:
    def test_with_reid_tracks_end_to_end(self, tmp_path: Path) -> None:
        from unittest.mock import Mock, patch

        from OTVision import dataformat
        from OTVision.track.tracker.tracker_plugin_botsort import BotsortTracker
        from tests.track.test_botsort_tracker import _create_botsort_track_config

        otdet = tmp_path / "clip.otdet"
        otdet.touch()
        write_video(otdet.with_suffix(".mp4"), 5)

        config = _create_botsort_track_config([otdet])
        config.botsort.tracker_params["with_reid"] = True
        config.botsort.tracker_params["model"] = "yolo11n-cls.pt"

        get_config = Mock()
        get_config.get.return_value = Mock(track=config)
        inner = BotsortTracker(get_current_config=get_config)
        tracker = VideoBackedTracker(
            inner, reid_model=config.botsort.tracker_params["model"]
        )
        id_gen = ids()

        fps_metadata = {dataformat.VIDEO: {dataformat.ACTUAL_FPS: 20.0}}
        with patch(
            "OTVision.track.tracker.tracker_plugin_botsort" ".read_json_bz2_metadata",
            return_value=fps_metadata,
        ):
            results = [
                tracker.track_frame(frame_for(no, otdet), id_gen) for no in range(1, 6)
            ]

        assert len(results) == 5
        tracked = [d for r in results for d in r.detections]
        assert tracked, "expected tracked detections with ReID enabled"
        assert all(r.image is None for r in results)


@pytest.mark.skipif(not ultralytics_available, reason="ultralytics required")
class TestBuilderWiring:
    def _builder_for(self, tracker_params_overrides: dict):
        from unittest.mock import Mock

        from OTVision.track.builder import TrackBuilder
        from tests.track.test_botsort_tracker import _create_botsort_track_config

        config = _create_botsort_track_config([])
        config.botsort.tracker_params.update(tracker_params_overrides)
        builder = TrackBuilder()
        builder.get_current_config = Mock()
        builder.get_current_config.get.return_value = Mock(track=config)
        return builder

    def test_with_reid_wraps_tracker_in_video_backed(self) -> None:
        builder = self._builder_for({"with_reid": True, "model": "yolo11n-cls.pt"})
        assert isinstance(builder.tracker._tracker, VideoBackedTracker)

    def test_without_reid_keeps_plain_botsort(self) -> None:
        from OTVision.track.tracker.tracker_plugin_botsort import BotsortTracker

        builder = self._builder_for({"with_reid": False})
        assert isinstance(builder.tracker._tracker, BotsortTracker)
        assert not isinstance(builder.tracker._tracker, VideoBackedTracker)

    def test_with_reid_model_auto_fails_fast(self) -> None:
        builder = self._builder_for({"with_reid": True, "model": "auto"})
        with pytest.raises(ValueError, match="auto"):
            builder.tracker
