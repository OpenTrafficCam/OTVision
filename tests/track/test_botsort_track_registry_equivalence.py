"""Differential test: ``TrackRegistry`` against the pre-refactor bookkeeping.

The BoT-SORT adapter originally kept two dicts keyed by OTVision track id
plus a ``TypedDict`` of lifecycle state, and inlined all of it into a ~135
line ``track_frame``. That was replaced by :class:`TrackRegistry`, keyed by
Ultralytics track id. This module pins the refactor by replaying both models
over randomized frame sequences and asserting identical observable output.

``_reference_model`` is a faithful transcription of the original logic (commit
de25329) and exists only as an oracle. Do not "improve" it; if it diverges
from ``TrackRegistry``, one of the two is wrong.
"""

import itertools
import random

import pytest

from OTVision.domain.tracker import TrackerLifecycle
from OTVision.track.tracker.tracker_plugin_botsort import TrackRegistry

Frames = list[tuple[int, list[int]]]
Result = list[tuple[list[tuple[int, int, bool]], frozenset, frozenset]]


def _reference_model(frames: Frames, t_min: int, t_miss_max: int) -> Result:
    """Replay the original two-map bookkeeping from commit de25329.

    Args:
        frames (Frames): ``(frame_no, ultralytics_track_ids)`` per frame.
        t_min (int): Minimum track span in frames.
        t_miss_max (int): Missing frames tolerated before finish/discard.

    Returns:
        Result: Per frame, the detection assignments and the finished and
        discarded OTVision track ids.
    """
    ids = itertools.count(1)
    botsort_to_ot: dict[int, int] = {}
    ot_to_botsort: dict[int, int] = {}
    state: dict[int, dict[str, int]] = {}
    out: Result = []

    for frame_no, botsort_ids in frames:
        det_to_ot: dict[int, int] = {}
        for det_idx, botsort_id in enumerate(botsort_ids):
            if botsort_id not in botsort_to_ot:
                ot_id = next(ids)
                botsort_to_ot[botsort_id] = ot_id
                ot_to_botsort[ot_id] = botsort_id
            else:
                ot_id = botsort_to_ot[botsort_id]
            det_to_ot[det_idx] = ot_id

            entry = state.get(ot_id)
            if entry is None:
                state[ot_id] = {
                    "first_frame": frame_no,
                    "last_frame": frame_no,
                    "age_missing": 0,
                }
            else:
                entry["last_frame"] = frame_no
                entry["age_missing"] = 0

        seen = set(det_to_ot.values())
        for ot_id, entry in list(state.items()):
            if ot_id not in seen:
                entry["age_missing"] += 1

        finished: set[int] = set()
        discarded: set[int] = set()
        for ot_id, entry in list(state.items()):
            if entry["age_missing"] <= t_miss_max:
                continue
            span = entry["last_frame"] - entry["first_frame"]
            (finished if span >= t_min else discarded).add(ot_id)
            mapped = ot_to_botsort.get(ot_id)
            if mapped is not None:
                botsort_to_ot.pop(mapped, None)
            ot_to_botsort.pop(ot_id, None)
            state.pop(ot_id, None)

        detections = [
            (
                det_idx,
                det_to_ot[det_idx],
                (
                    state[det_to_ot[det_idx]]["first_frame"] == frame_no
                    if det_to_ot[det_idx] in state
                    else True
                ),
            )
            for det_idx in range(len(botsort_ids))
            if det_idx in det_to_ot
        ]
        out.append((detections, frozenset(finished), frozenset(discarded)))
    return out


def _registry_model(frames: Frames, t_min: int, t_miss_max: int) -> Result:
    """Replay the same sequence through :class:`TrackRegistry`.

    Args:
        frames (Frames): ``(frame_no, ultralytics_track_ids)`` per frame.
        t_min (int): Minimum track span in frames.
        t_miss_max (int): Missing frames tolerated before finish/discard.

    Returns:
        Result: Same shape as :func:`_reference_model`.
    """
    ids = itertools.count(1)
    registry = TrackRegistry()
    lifecycle = TrackerLifecycle(t_min=t_min, t_miss_max=t_miss_max)
    out: Result = []

    for frame_no, botsort_ids in frames:
        assignments = {
            det_idx: registry.observe(botsort_id, frame_no, ids)
            for det_idx, botsort_id in enumerate(botsort_ids)
        }
        registry.age_unobserved(set(botsort_ids))
        finished, discarded = registry.evict_expired(lifecycle)
        detections = [
            (det_idx, a.ot_id, a.is_first) for det_idx, a in assignments.items()
        ]
        out.append((detections, frozenset(finished), frozenset(discarded)))
    return out


def _random_frames(rng: random.Random) -> Frames:
    """Build a random frame sequence of Ultralytics track ids.

    The small id pool makes reuse-after-eviction and two-detections-sharing-one-id
    common rather than rare.

    Args:
        rng (random.Random): Seeded generator.

    Returns:
        Frames: Randomized frame sequence.
    """
    return [
        (frame_no, [rng.randint(1, 4) for _ in range(rng.randint(0, 3))])
        for frame_no in range(1, rng.randint(1, 14) + 1)
    ]


@pytest.mark.parametrize("seed", range(400))
def test_registry_matches_pre_refactor_bookkeeping(seed: int) -> None:
    """The registry must be observationally identical to the original logic.

    Covers id reuse after eviction, several detections sharing one Ultralytics
    id within a frame, and gaps that trip the missing-frame threshold.
    """
    rng = random.Random(seed)
    frames = _random_frames(rng)
    t_min = rng.randint(0, 4)
    t_miss_max = rng.randint(0, 3)

    assert _registry_model(frames, t_min, t_miss_max) == _reference_model(
        frames, t_min, t_miss_max
    )
