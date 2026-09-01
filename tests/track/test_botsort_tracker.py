"""
Integration and unit tests for the BoT-SORT tracker adapter.
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

import importlib.util
import logging
import os
import shutil
from functools import cached_property
from pathlib import Path
from typing import Iterator
from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from OTVision import dataformat
from OTVision.application.config import (
    DEFAULT_BOTSORT_TRACKER_PARAMS,
    Config,
    TrackConfig,
    _TrackBotSortConfig,
    _TrackIouConfig,
)
from OTVision.application.track.botsort_params import (
    derive_track_buffer,
    extract_frame_rate_from_metadata,
    resolve_botsort_tracker_params,
    ultralytics_effective_miss_frames,
    validate_botsort_gmc_config,
    validate_botsort_reid_config,
)
from OTVision.application.track.tracker_metadata import tracker_metadata_of
from OTVision.application.track.tracking_run_id import StrIdGenerator
from OTVision.config import CONFIG
from OTVision.domain.detection import Detection
from OTVision.domain.frame import DetectedFrame
from OTVision.domain.tracker import TrackerLifecycle, TrackerType
from OTVision.track.tracker.tracker_plugin_botsort import (
    BotsortTracker,
    BoTSORTTrackerLike,
    TrackAssignment,
    TrackRegistry,
    UltralyticsResultsLike,
    validate_botsort_update_rows,
)
from tests.conftest import YieldFixture
from tests.track.helper.data_builder import create_frame

TEST_RUN_ID = "test-botsort-run"
_FPS_METADATA = {dataformat.VIDEO: {dataformat.ACTUAL_FPS: 20.0}}


def _create_botsort_track_config(
    paths: list | None = None,
    t_min: int = 5,
    t_miss_max: int = 51,
    overwrite: bool = True,
    tracker_params: dict | None = None,
) -> TrackConfig:
    """Build a BoT-SORT ``TrackConfig`` for tests."""
    params = dict(DEFAULT_BOTSORT_TRACKER_PARAMS)
    if tracker_params:
        params.update(tracker_params)
    return TrackConfig(
        paths=paths or [],
        run_chained=True,
        tracker_type=TrackerType.BOTSORT,
        botsort=_TrackBotSortConfig(
            t_min=t_min,
            t_miss_max=t_miss_max,
            tracker_params=params,
        ),
        overwrite=overwrite,
    )


def _make_detection(x: float = 100.0, y: float = 100.0, conf: float = 0.9) -> Detection:
    """Create a simple car detection for tracker tests."""
    return Detection(label="car", conf=conf, x=x, y=y, w=50.0, h=30.0)


def _make_frame(
    frame_no: int,
    source: str = "video1.otdet",
    detections: list[Detection] | None = None,
) -> DetectedFrame:
    """Create a ``DetectedFrame`` with optional detections."""
    if detections is None:
        detections = [_make_detection()]
    return create_frame(
        frame_number=frame_no,
        detections=detections,
        input_file_path=Path(source),
    )


def _mock_get_current_config(track_config: TrackConfig) -> Mock:
    """Wrap a track config in a ``GetCurrentConfig``-like mock."""
    config = Config(track=track_config)
    mock = Mock()
    mock.get.return_value = config
    return mock


def _botsort_row(
    det_idx: int,
    track_id: int,
    score: float = 0.9,
    x1: float = 75.0,
    y1: float = 85.0,
    x2: float = 125.0,
    y2: float = 115.0,
    cls: float = 0.0,
) -> list[float]:
    """Build an Nx8 Ultralytics BoT-SORT update row."""
    return [x1, y1, x2, y2, float(track_id), score, cls, float(det_idx)]


class FakeBoTSORT:
    """Deterministic stand-in for ultralytics BOTSORT."""

    def __init__(
        self, rows_by_frame: dict[int, list[list[float]]] | None = None
    ) -> None:
        """Store optional per-call result rows.

        Args:
            rows_by_frame: Mapping from zero-based call index to update rows.
        """
        self.rows_by_frame = rows_by_frame or {}
        self.call_count = 0

    def update(
        self,
        results: UltralyticsResultsLike,
        img: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.floating]:
        """Return scripted or default one-track-per-detection rows."""
        self.call_count += 1
        if self.call_count - 1 in self.rows_by_frame:
            rows = self.rows_by_frame[self.call_count - 1]
        else:
            rows = [
                _botsort_row(det_idx=i, track_id=i + 1, score=float(results.conf[i]))
                for i in range(len(results))
            ]
        if not rows:
            return np.zeros((0, 8), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)


class BadShapeBoTSORT:
    """BOTSORT stub that returns an invalid update shape."""

    def update(
        self,
        results: UltralyticsResultsLike,
        img: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.floating]:
        """Return an Nx4 array that must fail the shape guard."""
        return np.zeros((1, 4), dtype=np.float32)


# ---------------------------------------------------------------------------
# Pure unit tests (no ultralytics required)
# ---------------------------------------------------------------------------


def test_derive_track_buffer_aligns_occlusion_window_at_20fps() -> None:
    """At 20 fps, ceil derivation yields buffer 90 for T_MISS_MAX 60."""
    assert derive_track_buffer(t_miss_max=60, frame_rate=20) == 90
    assert ultralytics_effective_miss_frames(20, 90) == 60


def test_derive_track_buffer_covers_non_divisible_fps() -> None:
    """ceil derivation must keep effective lifetime >= T_MISS_MAX for all FPS."""
    t_miss_max = 60
    for fps in range(1, 61):
        buffer = derive_track_buffer(t_miss_max=t_miss_max, frame_rate=fps)
        effective = ultralytics_effective_miss_frames(fps, buffer)
        assert (
            effective >= t_miss_max
        ), f"fps={fps}: buffer={buffer} effective={effective} < {t_miss_max}"


def test_derive_track_buffer_at_29fps_does_not_undershoot() -> None:
    """Regression: round(60*30/29)=62 under-shoots; ceil must cover."""
    buffer = derive_track_buffer(t_miss_max=60, frame_rate=29)
    assert ultralytics_effective_miss_frames(29, buffer) >= 60
    assert buffer == 63


def test_validate_botsort_update_rows_accepts_exact_nx8() -> None:
    """Accept empty or exact Nx8 update layouts."""
    validate_botsort_update_rows(np.zeros((2, 8), dtype=np.float32))
    validate_botsort_update_rows(None)
    validate_botsort_update_rows(np.zeros((0, 8), dtype=np.float32))


def test_validate_botsort_update_rows_rejects_wider_or_narrower() -> None:
    """Reject layouts that are not exactly eight columns."""
    with pytest.raises(ValueError, match="Unexpected BoT-SORT update"):
        validate_botsort_update_rows(np.zeros((2, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="Unexpected BoT-SORT update"):
        validate_botsort_update_rows(np.zeros((2, 9), dtype=np.float32))
    with pytest.raises(ValueError, match="Unexpected BoT-SORT update"):
        validate_botsort_update_rows(np.zeros((8,), dtype=np.float32))


def test_extract_frame_rate_returns_none_for_missing_video_section() -> None:
    """Missing video section yields no FPS."""
    assert extract_frame_rate_from_metadata({}) is None


def test_extract_frame_rate_returns_none_for_zero_fps() -> None:
    """Zero FPS values are treated as missing."""
    metadata = {
        dataformat.VIDEO: {dataformat.ACTUAL_FPS: 0, dataformat.RECORDED_FPS: 0}
    }
    assert extract_frame_rate_from_metadata(metadata) is None


def test_botsort_metadata_omits_iou_sigma_fields() -> None:
    """BoT-SORT metadata keeps lifecycle params and omits unused IOU sigmas."""
    meta = tracker_metadata_of(
        _create_botsort_track_config(t_min=5, t_miss_max=60),
        {"match_thresh": 0.9},
    ).to_dict()
    assert meta[dataformat.NAME] == "BoTSORT"
    assert dataformat.SIGMA_L not in meta
    assert dataformat.SIGMA_H not in meta
    assert dataformat.SIGMA_IOU not in meta
    assert meta[dataformat.T_MIN] == 5
    assert meta[dataformat.T_MISS_MAX] == 60
    assert meta[dataformat.TRACKER_PARAMS] == {"match_thresh": 0.9}


def test_iou_metadata_keeps_sigma_fields() -> None:
    """IOU metadata retains sigma thresholds and omits BoT-SORT params."""
    meta = tracker_metadata_of(
        TrackConfig(
            iou=_TrackIouConfig(
                sigma_l=0.1, sigma_h=0.2, sigma_iou=0.3, t_min=5, t_miss_max=51
            ),
            tracker_type=TrackerType.IOU,
        )
    ).to_dict()
    assert meta[dataformat.NAME] == "IOU"
    assert meta[dataformat.SIGMA_L] == 0.1
    assert meta[dataformat.SIGMA_H] == 0.2
    assert meta[dataformat.SIGMA_IOU] == 0.3
    assert dataformat.TRACKER_PARAMS not in meta


def test_default_tracker_type_is_iou() -> None:
    """Application defaults keep IOU as the selected tracker."""
    assert TrackConfig().tracker_type == "iou"
    assert CONFIG["TRACK"]["TRACKER_TYPE"] == "iou"


def test_default_botsort_params_omit_track_buffer() -> None:
    """Defaults omit track_buffer so FPS derivation can run."""
    assert "track_buffer" not in DEFAULT_BOTSORT_TRACKER_PARAMS
    assert "track_buffer" not in _TrackBotSortConfig().tracker_params


def test_resolve_botsort_tracker_params_merges_defaults_and_derives_buffer() -> None:
    """Resolver merges defaults and derives track_buffer when unset."""
    config = _TrackBotSortConfig(
        t_min=5,
        t_miss_max=60,
        tracker_params={"gmc_method": "none", "with_reid": False},
    )
    resolved = resolve_botsort_tracker_params(config, frame_rate=20)
    assert resolved["track_high_thresh"] == 0.2
    assert resolved["match_thresh"] == 0.9
    assert resolved["gmc_method"] == "none"
    assert resolved["track_buffer"] == 90


def test_resolve_botsort_tracker_params_warns_on_undersized_explicit_buffer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit track_buffer below T_MISS_MAX logs a warning."""
    config = _TrackBotSortConfig(
        t_min=5,
        t_miss_max=60,
        tracker_params={"track_buffer": 30, "gmc_method": "none"},
    )
    with caplog.at_level(logging.WARNING):
        resolved = resolve_botsort_tracker_params(config, frame_rate=20)
    assert resolved["track_buffer"] == 30
    assert any("TRACK_BUFFER" in record.message for record in caplog.records)


def test_resolve_botsort_tracker_params_silent_on_oversized_explicit_buffer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An oversized window is harmless: OTVision still evicts at T_MISS_MAX."""
    config = _TrackBotSortConfig(
        t_min=5,
        t_miss_max=60,
        # At 20 fps: int(20/30 * 180) = 120 > T_MISS_MAX 60
        tracker_params={"track_buffer": 180, "gmc_method": "none"},
    )
    with caplog.at_level(logging.WARNING):
        resolved = resolve_botsort_tracker_params(config, frame_rate=20)
    assert resolved["track_buffer"] == 180
    assert not any("TRACK_BUFFER" in record.message for record in caplog.records)


def test_validate_botsort_reid_rejects_reid_entirely() -> None:
    """ReID is rejected: Ultralytics 8.3.159 disables its encoder regardless."""
    with pytest.raises(ValueError, match="ReID is not supported"):
        validate_botsort_reid_config({"with_reid": True})


def test_validate_botsort_reid_allows_reid_disabled() -> None:
    """Disabled ReID passes validation."""
    validate_botsort_reid_config({"with_reid": False})


def test_validate_botsort_gmc_rejects_non_none_method() -> None:
    """GMC needs frame images, which .otdet never carries."""
    with pytest.raises(ValueError, match="requires frame images"):
        validate_botsort_gmc_config({"gmc_method": "sparseOptFlow"})


def test_validate_botsort_gmc_allows_none() -> None:
    """``GMC_METHOD: none`` passes validation."""
    validate_botsort_gmc_config({"gmc_method": "none"})


def test_mocked_id_mapping_and_is_first() -> None:
    """Stable Ultralytics IDs map to one OTVision ID; first frame is flagged."""
    track_config = _create_botsort_track_config(t_miss_max=100)
    tracker = BotsortTracker(get_current_config=_mock_get_current_config(track_config))
    fake = FakeBoTSORT(
        rows_by_frame={
            0: [_botsort_row(det_idx=0, track_id=7)],
            1: [_botsort_row(det_idx=0, track_id=7)],
        }
    )
    id_gen: Iterator[int] = iter(range(1, 100))

    with (
        patch.object(tracker, "_ensure_botsort_initialized", return_value=fake),
        patch.object(tracker, "_reset_for_new_group"),
    ):
        first = tracker.track_frame(_make_frame(frame_no=0), id_gen)
        second = tracker.track_frame(_make_frame(frame_no=1), id_gen)

    assert len(first.detections) == 1
    assert first.detections[0].track_id == 1
    assert first.detections[0].is_first is True
    assert second.detections[0].track_id == 1
    assert second.detections[0].is_first is False


def test_mocked_t_min_discards_short_tracks() -> None:
    """Tracks shorter than t_min are discarded after exceeding t_miss_max misses."""
    track_config = _create_botsort_track_config(t_min=5, t_miss_max=2)
    tracker = BotsortTracker(get_current_config=_mock_get_current_config(track_config))
    fake = FakeBoTSORT(
        rows_by_frame={
            0: [_botsort_row(det_idx=0, track_id=1)],
            1: [_botsort_row(det_idx=0, track_id=1)],
            2: [],
            3: [],
            4: [],
        }
    )
    id_gen = iter(range(1, 100))
    discarded: set[int] = set()
    finished: set[int] = set()

    with (
        patch.object(tracker, "_ensure_botsort_initialized", return_value=fake),
        patch.object(tracker, "_reset_for_new_group"),
    ):
        for frame_no in range(5):
            result = tracker.track_frame(
                _make_frame(
                    frame_no=frame_no,
                    detections=[_make_detection()] if frame_no < 2 else [],
                ),
                id_gen,
            )
            discarded |= result.discarded_tracks
            finished |= result.finished_tracks

    assert discarded == {1}
    assert finished == set()


def test_mocked_t_miss_max_finishes_long_tracks() -> None:
    """Tracks spanning at least t_min finish after exceeding t_miss_max misses."""
    track_config = _create_botsort_track_config(t_min=2, t_miss_max=2)
    tracker = BotsortTracker(get_current_config=_mock_get_current_config(track_config))
    fake = FakeBoTSORT(
        rows_by_frame={
            0: [_botsort_row(det_idx=0, track_id=1)],
            1: [_botsort_row(det_idx=0, track_id=1)],
            2: [_botsort_row(det_idx=0, track_id=1)],
            3: [_botsort_row(det_idx=0, track_id=1)],
            4: [],
            5: [],
            6: [],
        }
    )
    id_gen = iter(range(1, 100))
    finished: set[int] = set()

    with (
        patch.object(tracker, "_ensure_botsort_initialized", return_value=fake),
        patch.object(tracker, "_reset_for_new_group"),
    ):
        for frame_no in range(7):
            dets = [_make_detection()] if frame_no < 4 else []
            result = tracker.track_frame(
                _make_frame(frame_no=frame_no, detections=dets), id_gen
            )
            finished |= result.finished_tracks

    assert finished == {1}


def test_explicit_reset_clears_id_maps() -> None:
    """``reset()`` clears Ultralytics state and OTVision ID maps."""
    track_config = _create_botsort_track_config(t_miss_max=100)
    tracker = BotsortTracker(get_current_config=_mock_get_current_config(track_config))
    fake: BoTSORTTrackerLike = FakeBoTSORT()
    tracker._botsort = fake
    tracker._registry.observe(99, frame_no=1, id_generator=iter([1]))

    tracker.reset()

    assert tracker._botsort is None
    assert tracker._registry._entries == {}


def test_shape_guard_raised_during_track_frame() -> None:
    """Invalid update shapes raise during ``track_frame``."""
    track_config = _create_botsort_track_config(t_miss_max=100)
    tracker = BotsortTracker(get_current_config=_mock_get_current_config(track_config))

    with (
        patch.object(
            tracker, "_ensure_botsort_initialized", return_value=BadShapeBoTSORT()
        ),
        patch.object(tracker, "_reset_for_new_group"),
    ):
        with pytest.raises(ValueError, match="Unexpected BoT-SORT update"):
            tracker.track_frame(_make_frame(frame_no=1), iter(range(1, 10)))


def test_build_args_rejects_reid() -> None:
    """Tracker construction rejects WITH_REID."""
    track_config = TrackConfig(
        tracker_type=TrackerType.BOTSORT,
        botsort=_TrackBotSortConfig(
            t_min=5,
            t_miss_max=60,
            tracker_params={"with_reid": True},
        ),
    )
    tracker = BotsortTracker(get_current_config=_mock_get_current_config(track_config))
    with pytest.raises(ValueError, match="ReID is not supported"):
        tracker._build_args(frame_rate=20)


def test_fps_extraction_non_otdet_source_raises() -> None:
    """Non-.otdet source file must raise ValueError."""
    track_config = _create_botsort_track_config()
    tracker = BotsortTracker(get_current_config=_mock_get_current_config(track_config))
    frame = _make_frame(frame_no=0, source="video.mp4")

    with pytest.raises(ValueError, match="requires FPS metadata from an .otdet"):
        tracker.track_frame(frame, iter(range(1, 100)))


@patch(
    "OTVision.track.tracker.tracker_plugin_botsort.read_json_bz2_metadata",
    side_effect=OSError("unreadable"),
)
def test_fps_extraction_unreadable_file_raises(_mock: Mock) -> None:
    """Unreadable .otdet file must raise ValueError."""
    track_config = _create_botsort_track_config()
    tracker = BotsortTracker(get_current_config=_mock_get_current_config(track_config))
    frame = _make_frame(frame_no=0, source="video.otdet")

    with pytest.raises(ValueError, match="readable .otdet metadata"):
        tracker.track_frame(frame, iter(range(1, 100)))


@patch(
    "OTVision.track.tracker.tracker_plugin_botsort.read_json_bz2_metadata",
    return_value={},
)
def test_fps_extraction_missing_fps_keys_raises(_mock: Mock) -> None:
    """Missing FPS keys in metadata must raise ValueError."""
    track_config = _create_botsort_track_config()
    tracker = BotsortTracker(get_current_config=_mock_get_current_config(track_config))
    frame = _make_frame(frame_no=0, source="video.otdet")

    with pytest.raises(ValueError, match="requires FPS metadata in .otdet"):
        tracker.track_frame(frame, iter(range(1, 100)))


# ---------------------------------------------------------------------------
# Integration tests requiring ultralytics
# ---------------------------------------------------------------------------


def _ultralytics_available() -> bool:
    """Return whether the optional ultralytics dependency is installed."""
    return importlib.util.find_spec("ultralytics") is not None


requires_ultralytics = pytest.mark.skipif(
    not _ultralytics_available(), reason="ultralytics not installed"
)


@requires_ultralytics
class TestBotsortUltralyticsIntegration:
    """End-to-end BoT-SORT tests against the pinned ultralytics package."""

    @pytest.fixture(autouse=True)
    def _mock_versions(self) -> None:
        """Ignore version strings in generated metadata."""
        from OTVision import version

        version.otvision_version = Mock(return_value="ignored")
        version.ottrack_version = Mock(return_value="ignored")
        version.otdet_version = Mock(return_value="ignored")

    @pytest.fixture
    def test_track_dir(self, test_data_dir: Path) -> Path:
        """Return the fixture directory containing track test data."""
        return test_data_dir / "track"

    @pytest.fixture
    def test_track_tmp_dir(
        self, test_data_tmp_dir: Path, test_track_dir: Path
    ) -> YieldFixture[Path]:
        """Copy track fixtures into a writable temporary directory."""
        tmp = test_data_tmp_dir / "track_botsort"
        tmp.mkdir(exist_ok=True)
        shutil.copytree(test_track_dir, tmp, dirs_exist_ok=True)
        extension = CONFIG["DEFAULT_FILETYPE"]["TRACK"]
        for f in tmp.rglob(f"*{extension}"):
            f.unlink()
        yield tmp
        shutil.rmtree(tmp)

    @pytest.mark.asyncio
    async def test_botsort_tracking_produces_ottrk_output(
        self,
        test_track_tmp_dir: Path,
    ) -> None:
        """Run BoT-SORT on real .otdet fixtures and expect .ottrk output."""
        from OTVision.track.builder import TrackBuilder

        class MockBotsortTrackBuilder(TrackBuilder):
            @cached_property
            def tracking_run_id_generator(self) -> StrIdGenerator:
                return lambda: TEST_RUN_ID

        input_folder = test_track_tmp_dir / "default"
        track_config = _create_botsort_track_config(
            paths=[str(input_folder.relative_to(os.getcwd()))],
            tracker_params={
                "gmc_method": "none",
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.25,
                "with_reid": False,
            },
        )
        builder = MockBotsortTrackBuilder()
        builder.update_current_track_config.update(track_config)
        otvision_track = builder.build()

        await otvision_track.start()

        extension = CONFIG["DEFAULT_FILETYPE"]["TRACK"]
        output_files = list(input_folder.glob(f"*{extension}"))
        assert len(output_files) > 0, "No .ottrk files generated by BoT-SORT"

    @patch(
        "OTVision.track.tracker.tracker_plugin_botsort.read_json_bz2_metadata",
        return_value=_FPS_METADATA,
    )
    def test_botsort_resets_state_on_new_video_group(
        self,
        _mock_metadata: Mock,
    ) -> None:
        """An explicit reset at a group boundary must not leak track ids."""
        track_config = _create_botsort_track_config(t_miss_max=100)
        tracker = BotsortTracker(
            get_current_config=_mock_get_current_config(track_config)
        )

        id_gen = iter(range(1, 10000))

        group1_ids: set[int] = set()
        for i in range(5):
            result = tracker.track_frame(
                _make_frame(frame_no=i, source="v1.otdet"), id_gen
            )
            for det in result.detections:
                group1_ids.add(det.track_id)

        assert len(group1_ids) > 0, "No tracks assigned in group 1"

        # GroupedFilesTracker resets explicitly at each group boundary.
        tracker.reset()

        group2_ids: set[int] = set()
        for i in range(5):
            result = tracker.track_frame(
                _make_frame(frame_no=i, source="v2.otdet"), id_gen
            )
            for det in result.detections:
                group2_ids.add(det.track_id)

        assert len(group2_ids) > 0, "No tracks assigned in group 2"
        assert group1_ids.isdisjoint(
            group2_ids
        ), f"Track IDs leaked across video groups: {group1_ids & group2_ids}"


class TestTrackRegistry:
    """Lifecycle behaviour of the registry that owns BoT-SORT id mapping."""

    def test_first_sighting_is_flagged_then_not(self) -> None:
        """An Ultralytics id maps to one OT id; only its first frame is first."""
        registry = TrackRegistry()
        ids = iter(range(1, 100))

        first = registry.observe(7, frame_no=1, id_generator=ids)
        second = registry.observe(7, frame_no=2, id_generator=ids)

        assert first == TrackAssignment(ot_id=1, is_first=True)
        assert second == TrackAssignment(ot_id=1, is_first=False)

    def test_distinct_ultralytics_ids_get_distinct_ot_ids(self) -> None:
        """Separate Ultralytics tracks never share an OTVision id."""
        registry = TrackRegistry()
        ids = iter(range(1, 100))

        assert registry.observe(7, 1, ids).ot_id != registry.observe(8, 1, ids).ot_id

    def test_track_survives_exactly_t_miss_max_misses(self) -> None:
        """Eviction happens only after MORE than t_miss_max consecutive misses."""
        registry = TrackRegistry()
        lifecycle = TrackerLifecycle(t_min=0, t_miss_max=3)
        registry.observe(7, frame_no=1, id_generator=iter([1]))

        for _ in range(lifecycle.t_miss_max):
            registry.age_unobserved(observed_botsort_track_ids=set())
            assert registry.evict_expired(lifecycle) == (set(), set())

        registry.age_unobserved(observed_botsort_track_ids=set())
        assert registry.evict_expired(lifecycle) == ({1}, set())

    def test_short_track_is_discarded_not_finished(self) -> None:
        """A track spanning fewer than t_min frames is discarded."""
        registry = TrackRegistry()
        lifecycle = TrackerLifecycle(t_min=5, t_miss_max=0)
        registry.observe(7, frame_no=1, id_generator=iter([1]))

        registry.age_unobserved(observed_botsort_track_ids=set())
        assert registry.evict_expired(lifecycle) == (set(), {1})

    def test_long_track_is_finished(self) -> None:
        """A track spanning at least t_min frames is finished."""
        registry = TrackRegistry()
        lifecycle = TrackerLifecycle(t_min=5, t_miss_max=0)
        ids = iter(range(1, 100))
        registry.observe(7, frame_no=1, id_generator=ids)
        registry.observe(7, frame_no=6, id_generator=ids)

        registry.age_unobserved(observed_botsort_track_ids=set())
        assert registry.evict_expired(lifecycle) == ({1}, set())

    def test_observation_resets_the_missing_counter(self) -> None:
        """Seeing a track again clears accumulated misses."""
        registry = TrackRegistry()
        lifecycle = TrackerLifecycle(t_min=0, t_miss_max=2)
        ids = iter(range(1, 100))
        registry.observe(7, frame_no=1, id_generator=ids)

        registry.age_unobserved(observed_botsort_track_ids=set())
        registry.age_unobserved(observed_botsort_track_ids=set())
        registry.observe(7, frame_no=4, id_generator=ids)
        registry.age_unobserved(observed_botsort_track_ids=set())

        assert registry.evict_expired(lifecycle) == (set(), set())
