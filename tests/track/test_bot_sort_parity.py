import json
import os
from dataclasses import dataclass, replace
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from OTVision.application.config import Config, TrackConfig
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.dataformat import DATA, DETECTIONS, METADATA
from OTVision.domain.current_config import CurrentConfig
from OTVision.domain.detection import Detection
from OTVision.domain.frame import DetectedFrame
from OTVision.helpers.files import denormalize_bbox, read_json
from OTVision.track.parser.chunk_parser_plugins import DetectionParser
from OTVision.track.tracker.bot_sort.tracker import BotSortTracker
from tests.track.test_bot_sort_tracker import detected_frame, detection

_RECORDINGS = {
    f"{kind}-{scene.lower()}": directory
    / f"Testvideo_Cars-{scene}_FR20_2020-01-01_00-00-00.otdet"
    for kind, directory in (
        ("root", Path()),
        ("detect", Path("detect/default")),
        ("track", Path("track/default")),
    )
    for scene in ("Cyclist", "Truck")
}
# The other IOU parameter directories contain byte-identical copies of track/default.
_MISS_LIMIT = 60
_CAR = detection()


def _crossing() -> list[tuple[Detection, ...]]:
    frames: list[tuple[Detection, ...]] = []
    for i in range(10):
        east = detection(32 + 4 * i)
        west = replace(detection(68 - 4 * i), y=54)
        frames.append((east, west) if i % 2 == 0 else (west, east))
    return frames


def _occlusion(missed_frames: int) -> list[tuple[Detection, ...]]:
    frames: list[tuple[Detection, ...]] = [(_CAR,)]
    frames.extend([()] * missed_frames)
    frames.extend([(_CAR,), (_CAR,)])
    return frames


_SCRIPTED: dict[str, list[tuple[Detection, ...]]] = {
    "crossing": _crossing(),
    "gap-under": _occlusion(_MISS_LIMIT - 1),
    "gap-exact": _occlusion(_MISS_LIMIT),
    "gap-over": _occlusion(_MISS_LIMIT + 1),
    "low-confidence": [
        (_CAR,),
        (detection(90, confidence=0.14), detection(52, confidence=0.15)),
        (_CAR,),
    ],
    "false-positive": [(_CAR,), (detection(90), _CAR), (_CAR,), (_CAR,)],
}
_CASES = [*_RECORDINGS, *_SCRIPTED]


def _frames(case: str, data_dir: Path) -> list[DetectedFrame]:
    if case in _SCRIPTED:
        return [
            detected_frame(no, *detections)
            for no, detections in enumerate(_SCRIPTED[case])
        ]

    path = data_dir / _RECORDINGS[case]
    data = read_json(path)
    denormalize_bbox(data, path, metadata={path.as_posix(): data[METADATA]})
    parser = DetectionParser()
    return [
        detected_frame(int(no), *parser.convert(frame[DETECTIONS]))
        for no, frame in sorted(data[DATA].items(), key=lambda item: int(item[0]))
    ]


@pytest.mark.parametrize("case", _CASES)
def test_matches_oracle_grouping(case: str, test_data_dir: Path) -> None:
    source = _frames(case, test_data_dir)
    path = test_data_dir / "bot_sort_parity" / f"{case}.json"
    reference: list[list[int | None]] = json.loads(path.read_text(encoding="utf-8"))
    tracker = BotSortTracker(GetCurrentConfig(CurrentConfig(Config())))
    ids = count(1)
    seen_ids: set[int] = set()
    actual_to_expected: dict[int, int] = {}
    expected_to_actual: dict[int, int] = {}

    for frame, expected_ids in zip(source, reference, strict=True):
        assert len(expected_ids) == len(frame.detections)
        # Labels identify input detections even when two have the same box and score.
        indexed = replace(
            frame,
            detections=[
                replace(d, label=str(i)) for i, d in enumerate(frame.detections)
            ],
        )
        output = tracker.track_frame(indexed, ids)
        expected = {
            i: track for i, track in enumerate(expected_ids) if track is not None
        }
        actual = {int(d.label): d for d in output.detections}
        assigned = {i: d.track_id for i, d in actual.items()}
        context = f"{case} frame {frame.no}: oracle {expected}, OTVision {assigned}"

        assert len(actual) == len(output.detections), context
        for index, tracked in actual.items():
            assert 0 <= index < len(frame.detections), context
            track = tracked.track_id
            assert tracked.is_first == (track not in seen_ids), context
            seen_ids.add(track)
            if index not in expected:
                assert tracked.is_first, context
        for index, expected_id in expected.items():
            assert index in actual, context
            actual_id = actual[index].track_id
            assert (
                actual_to_expected.setdefault(actual_id, expected_id) == expected_id
            ), context
            assert expected_to_actual.setdefault(expected_id, actual_id) == actual_id, (
                context
            )


@dataclass
class _OracleDetections:
    xywh: NDArray[np.float32]
    conf: NDArray[np.float32]
    cls: NDArray[np.int32]

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, indices: NDArray[np.bool_]) -> "_OracleDetections":
        return _OracleDetections(
            self.xywh[indices], self.conf[indices], self.cls[indices]
        )

    @property
    def xyxy(self) -> NDArray[np.float32]:
        centers = self.xywh[:, :2]
        half_sizes = self.xywh[:, 2:] / 2
        return np.concatenate((centers - half_sizes, centers + half_sizes), axis=1)


@pytest.mark.skipif(
    os.environ.get("UPDATE_BOT_SORT_PARITY") != "1",
    reason="Enable UPDATE_BOT_SORT_PARITY=1 to regenerate the oracle fixtures",
)
@pytest.mark.parametrize("case", _CASES)
def test_update_bot_sort_parity(case: str, test_data_dir: Path) -> None:
    """Use uv run --extra inference_cpu --with lap==0.5.12 pytest."""
    from ultralytics import __version__
    from ultralytics.trackers import BOTSORT

    assert __version__ == "8.3.159"
    settings = TrackConfig.bot_sort
    args = SimpleNamespace(
        tracker_type="botsort",
        track_high_thresh=settings.track_high_thresh,
        track_low_thresh=settings.track_low_thresh,
        new_track_thresh=settings.new_track_thresh,
        match_thresh=settings.match_thresh,
        fuse_score=settings.fuse_score,
        # At 30 fps, an isolated object revived after 60 empty frames with
        # buffer 59 and started a different ID after 61. Buffer 60 revived at 61.
        # Known difference, not covered by fixtures: a track the oracle revives
        # after exactly T_MISS_MAX missed frames gets a new ID from the oracle
        # the next time it is lost, even for one frame. OTVision keeps the ID.
        track_buffer=settings.t_miss_max - 1,
        gmc_method="none",
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        with_reid=False,
        model="auto",
    )
    tracker = BOTSORT(args=args, frame_rate=30)

    rows = []
    for frame in _frames(case, test_data_dir):
        detections = frame.detections
        # Return the input index in the class column even when confidence
        # bucketing changes the detection's position in the oracle's input.
        results = _OracleDetections(
            xywh=np.array(
                [(d.x, d.y, d.w, d.h) for d in detections], dtype=np.float32
            ).reshape(-1, 4),
            conf=np.array([d.conf for d in detections], dtype=np.float32),
            cls=np.arange(len(detections), dtype=np.int32),
        )
        assignments: list[int | None] = [None] * len(detections)
        for track in tracker.update(results):
            track_id, index = int(track[4]), int(track[6])
            assert assignments[index] is None
            assignments[index] = track_id
        rows.append(json.dumps(assignments, separators=(",", ":")))

    directory = test_data_dir / "bot_sort_parity"
    directory.mkdir(exist_ok=True)
    (directory / f"{case}.json").write_text(
        "[\n  " + ",\n  ".join(rows) + "\n]\n", encoding="utf-8"
    )
