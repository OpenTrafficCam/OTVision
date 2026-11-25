from dataclasses import replace
from datetime import datetime
from itertools import count

import pytest

from OTVision.application.config import Config, TrackConfig
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.domain.current_config import CurrentConfig
from OTVision.domain.detection import Detection
from OTVision.domain.frame import DetectedFrame
from OTVision.track.tracker.bot_sort.tracker import BotSortTracker

DEFAULTS = TrackConfig.bot_sort
HIGH_CONFIDENCE = 0.9
"""Above NEW_TRACK_THRESH, so an unmatched detection starts a track."""
LOW_CONFIDENCE = 0.15
"""Between TRACK_LOW_THRESH and TRACK_HIGH_THRESH: it can extend a track that is
already being tracked, but never start one."""
assert DEFAULTS.track_low_thresh < LOW_CONFIDENCE < DEFAULTS.track_high_thresh
assert HIGH_CONFIDENCE >= DEFAULTS.new_track_thresh


def detection(
    x: float = 50, confidence: float = HIGH_CONFIDENCE, label: str = "car"
) -> Detection:
    """Return a 20 x 20 box centered at (x, 50).

    Two such boxes 3 apart in x have an IoU of 340/460 = 0.74, and 1 apart an IoU
    of 380/420 = 0.90.
    """
    return Detection(label, confidence, x, 50, 20, 20)


def detected_frame(no: int, *detections: Detection) -> DetectedFrame:
    return DetectedFrame(
        no=no,
        occurrence=datetime(2020, 1, 1),
        source="source",
        output="output",
        detections=detections,
    )


def test_emits_first_detection_with_supplied_id() -> None:
    tracker = BotSortTracker(GetCurrentConfig(CurrentConfig(Config())))
    first = detection()
    frame = tracker.track_frame(detected_frame(42, first), count(101))

    assert list(frame.detections) == [first.of_track(101, True)]
    assert frame.finished_tracks == frame.discarded_tracks == set()


def test_matches_high_confidence_and_keeps_raw_box_and_label() -> None:
    tracker = BotSortTracker(GetCurrentConfig(CurrentConfig(Config())))
    ids = count(101)
    tracker.track_frame(detected_frame(0, detection()), ids)
    moved = Detection("truck", 0.8, 53.123456789, 48.7, 22.8, 19.2)

    frame = tracker.track_frame(detected_frame(1, moved), ids)

    assert list(frame.detections) == [moved.of_track(101, False)]


def test_unconfirmed_track_is_discarded_if_next_frame_misses() -> None:
    tracker = BotSortTracker(GetCurrentConfig(CurrentConfig(Config())))
    ids = count(101)
    tracker.track_frame(detected_frame(0), ids)
    first = detection()
    born = tracker.track_frame(detected_frame(1, first), ids)

    missed = tracker.track_frame(detected_frame(2), ids)
    returned = tracker.track_frame(detected_frame(3, first), ids)

    assert list(born.detections) == [first.of_track(101, True)]
    assert missed.discarded_tracks == {101}
    assert missed.finished_tracks == set()
    assert list(returned.detections) == [first.of_track(102, True)]


def test_low_confidence_extends_confirmed_track_without_starting_one() -> None:
    tracker = BotSortTracker(GetCurrentConfig(CurrentConfig(Config())))
    ids = count(101)
    tracker.track_frame(detected_frame(0, detection()), ids)
    low = detection(confidence=LOW_CONFIDENCE)

    frame = tracker.track_frame(
        detected_frame(1, low, detection(x=100, confidence=LOW_CONFIDENCE)), ids
    )

    assert list(frame.detections) == [low.of_track(101, False)]


def test_first_frame_track_revives_at_miss_limit() -> None:
    tracker = BotSortTracker(GetCurrentConfig(CurrentConfig(Config())))
    ids = count(101)
    car = detection()
    first_no = 42
    tracker.track_frame(detected_frame(first_no, car), ids)
    missed_nos = range(first_no + 1, first_no + 1 + DEFAULTS.t_miss_max)
    for no in missed_nos:
        missed = tracker.track_frame(detected_frame(no), ids)
        assert missed.finished_tracks == missed.discarded_tracks == set()

    returned = tracker.track_frame(detected_frame(missed_nos.stop, car), ids)

    assert list(returned.detections) == [car.of_track(101, False)]


def test_above_miss_limit_ends_old_track_once_and_starts_new_id() -> None:
    tracker = BotSortTracker(GetCurrentConfig(CurrentConfig(Config())))
    ids = count(101)
    car = detection()
    # A span of T_MIN, so the track is reported as finished, not discarded.
    seen_nos = range(DEFAULTS.t_min + 1)
    for no in seen_nos:
        tracker.track_frame(detected_frame(no, car), ids)
    missed_nos = range(seen_nos.stop, seen_nos.stop + DEFAULTS.t_miss_max)
    for no in missed_nos:
        missed = tracker.track_frame(detected_frame(no), ids)
        assert missed.finished_tracks == missed.discarded_tracks == set()

    expired = tracker.track_frame(detected_frame(missed_nos.stop), ids)
    returned = tracker.track_frame(detected_frame(missed_nos.stop + 1, car), ids)

    assert expired.finished_tracks == {101}
    assert expired.discarded_tracks == set()
    assert list(returned.detections) == [car.of_track(102, True)]
    assert returned.finished_tracks == returned.discarded_tracks == set()


@pytest.mark.parametrize(
    "birth, finished, discarded, survivor",
    [(6, set(), {102}, 101), (2, {102}, set(), 101), (1, {101}, set(), 102)],
    ids=["shorter-tracked", "tie", "shorter-lost"],
)
def test_duplicate_ends_shorter_track_and_suppresses_its_detection(
    birth: int, finished: set[int], discarded: set[int], survivor: int
) -> None:
    """Old (101) is seen in frames 0-6, a span of 6, and young (102) from `birth`
    to 8. In frame 8 a low-confidence detection pulls young onto the lost old
    track, and the shorter of the two ends. On a tie, the tracked one ends.

    - birth 6: young spans 2, ends, and is discarded (below T_MIN)
    - birth 2: young spans 6 (tie), ends, and is finished
    - birth 1: young spans 7, so old ends and is finished
    """
    # Matching needs an IoU above 0.9, so the boxes 3 apart (IoU 0.74) stay
    # separate tracks.
    settings = replace(TrackConfig.bot_sort, match_thresh=0.1, fuse_score=False)
    tracker = BotSortTracker(
        GetCurrentConfig(CurrentConfig(Config(track=TrackConfig(bot_sort=settings))))
    )
    ids = count(101)
    old = detection()
    young = detection(x=53)
    for no in range(7):
        detections = (old,) if no < birth else (old, young)
        tracker.track_frame(detected_frame(no, *detections), ids)
    confirmed = tracker.track_frame(detected_frame(7, young), ids)

    low = detection(confidence=LOW_CONFIDENCE)
    duplicate = tracker.track_frame(detected_frame(8, low), ids)
    revived = tracker.track_frame(detected_frame(9, old), ids)

    assert list(confirmed.detections) == [young.of_track(102, False)]
    assert list(duplicate.detections) == (
        [low.of_track(102, False)] if survivor == 102 else []
    )
    assert duplicate.finished_tracks == finished
    assert duplicate.discarded_tracks == discarded
    assert list(revived.detections) == [old.of_track(survivor, False)]
    assert revived.finished_tracks == revived.discarded_tracks == set()


def test_reset_clears_tracked_lost_and_unconfirmed_ids_and_restarts_confirmation() -> (
    None
):
    tracker = BotSortTracker(GetCurrentConfig(CurrentConfig(Config())))
    ids = count(101)
    car, truck, bicycle = detection(), detection(x=100), detection(x=150)
    tracker.track_frame(detected_frame(0, car, truck), ids)
    tracker.track_frame(detected_frame(1, car, bicycle), ids)

    tracker.reset()
    first = tracker.track_frame(detected_frame(42, car, truck, bicycle), ids)
    missed = tracker.track_frame(detected_frame(43), ids)
    returned = tracker.track_frame(detected_frame(44, car, truck, bicycle), ids)

    assert list(first.detections) == [
        car.of_track(104, True),
        truck.of_track(105, True),
        bicycle.of_track(106, True),
    ]
    assert missed.finished_tracks == missed.discarded_tracks == set()
    assert list(returned.detections) == [
        car.of_track(104, False),
        truck.of_track(105, False),
        bicycle.of_track(106, False),
    ]
    assert first.finished_tracks == first.discarded_tracks == set()
    assert returned.finished_tracks == returned.discarded_tracks == set()


def test_expiry_reports_each_id_by_span_with_no_finish_confidence_gate() -> None:
    settings = replace(TrackConfig.bot_sort, t_min=2, t_miss_max=0)
    tracker = BotSortTracker(
        GetCurrentConfig(CurrentConfig(Config(track=TrackConfig(bot_sort=settings))))
    )
    ids = count(101)
    # Just above TRACK_HIGH_THRESH: finishing a track has no confidence gate.
    car, truck = detection(confidence=0.25), detection(x=100)
    first = tracker.track_frame(detected_frame(0, car, truck), ids)
    second = tracker.track_frame(detected_frame(1, car), ids)
    third = tracker.track_frame(detected_frame(2, car), ids)
    fourth = tracker.track_frame(detected_frame(3), ids)
    fifth = tracker.track_frame(detected_frame(4), ids)

    assert list(first.detections) == [
        car.of_track(101, True),
        truck.of_track(102, True),
    ]
    assert (
        list(second.detections) == list(third.detections) == [car.of_track(101, False)]
    )
    assert second.finished_tracks == set()
    assert second.discarded_tracks == {102}
    assert fourth.finished_tracks == {101}
    assert fourth.discarded_tracks == set()
    for frame in (first, third, fifth):
        assert frame.finished_tracks == frame.discarded_tracks == set()


def test_reads_current_config_while_tracking() -> None:
    current = CurrentConfig(Config())
    tracker = BotSortTracker(GetCurrentConfig(current))
    ids = count(101)
    car = detection()
    tracker.track_frame(detected_frame(0, car), ids)
    tracker.track_frame(detected_frame(1), ids)
    # The car is lost after one missed frame and cannot start a new track.
    settings = replace(
        TrackConfig.bot_sort, t_miss_max=0, new_track_thresh=HIGH_CONFIDENCE + 0.05
    )
    current.update(Config(track=TrackConfig(bot_sort=settings)))

    expired = tracker.track_frame(detected_frame(2, car), ids)

    assert list(expired.detections) == []
    assert expired.discarded_tracks == {101}
    assert expired.finished_tracks == set()


def test_new_track_ended_by_duplicate_removal_is_never_reported() -> None:
    # Matching needs an IoU above 0.95, so a box 1 away (IoU 0.90) starts a new
    # track instead. That still counts as a duplicate (IoU above 0.85).
    settings = replace(TrackConfig.bot_sort, match_thresh=0.05, fuse_score=False)
    tracker = BotSortTracker(
        GetCurrentConfig(CurrentConfig(Config(track=TrackConfig(bot_sort=settings))))
    )
    ids = count(101)
    car = detection()
    tracker.track_frame(detected_frame(0, car), ids)
    tracker.track_frame(detected_frame(1), ids)

    duplicate = tracker.track_frame(detected_frame(2, detection(x=51)), ids)
    revived = tracker.track_frame(detected_frame(3, car), ids)

    assert list(duplicate.detections) == []
    assert duplicate.finished_tracks == duplicate.discarded_tracks == set()
    assert list(revived.detections) == [car.of_track(101, False)]
    assert revived.finished_tracks == revived.discarded_tracks == set()
