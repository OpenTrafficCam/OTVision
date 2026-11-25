from dataclasses import replace
from datetime import datetime
from itertools import count

import pytest

from OTVision.application.config import TRACKER_TYPES, Config, TrackConfig
from OTVision.application.config_parser import InvalidOtvisionConfigError
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.domain.current_config import CurrentConfig
from OTVision.domain.detection import Detection
from OTVision.domain.frame import DetectedFrame, TrackedFrame
from OTVision.track.model.tracking_interfaces import IdGenerator, Tracker
from OTVision.track.tracker import configured_tracker
from OTVision.track.tracker.configured_tracker import create_tracker


@pytest.fixture
def detected_frame() -> DetectedFrame:
    return DetectedFrame(
        no=1,
        occurrence=datetime(2026, 1, 1),
        source="camera.mp4",
        output="camera.mp4",
        detections=[Detection(label="car", conf=0.8, x=50, y=50, w=20, h=10)],
    )


def test_create_tracker_resolves_type_on_first_use(
    detected_frame: DetectedFrame,
) -> None:
    current = CurrentConfig(Config(track=TrackConfig(tracker_type="unknown")))
    tracker = create_tracker(GetCurrentConfig(current))
    current.update(Config())

    tracked = tracker.track_frame(detected_frame, count(1))

    assert tracked.observed_tracks == {1}


def test_reset_resolves_current_type_and_resets_the_tracker(
    detected_frame: DetectedFrame,
) -> None:
    current = CurrentConfig(Config())
    tracker = create_tracker(GetCurrentConfig(current))
    ids = count(1)
    first = tracker.track_frame(detected_frame, ids)
    current.update(Config(track=TrackConfig(tracker_type="unknown")))

    with pytest.raises(InvalidOtvisionConfigError, match=r"'unknown'.*options: iou"):
        tracker.reset()

    current.update(Config())
    tracker.reset()
    second = tracker.track_frame(replace(detected_frame, no=2), ids)

    assert first.observed_tracks == {1}
    assert second.observed_tracks == {2}


def test_type_change_takes_effect_at_reset_not_between_frames(
    detected_frame: DetectedFrame,
) -> None:
    current = CurrentConfig(Config())
    tracker = create_tracker(GetCurrentConfig(current))
    ids = count(1)
    tracker.track_frame(detected_frame, ids)
    current.update(Config(track=TrackConfig(tracker_type="unknown")))

    second = tracker.track_frame(replace(detected_frame, no=2), ids)

    assert second.observed_tracks == {1}


def test_reset_reaches_the_selected_tracker(
    detected_frame: DetectedFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class RecordingTracker(Tracker):
        def __init__(self, get_current_config: GetCurrentConfig) -> None:
            pass

        def reset(self) -> None:
            calls.append("reset")

        def track_frame(
            self, frame: DetectedFrame, id_generator: IdGenerator
        ) -> TrackedFrame:
            calls.append("frame")
            return TrackedFrame(
                no=frame.no,
                occurrence=frame.occurrence,
                source=frame.source,
                output=frame.output,
                detections=[],
                image=frame.image,
                finished_tracks=set(),
                discarded_tracks=set(),
            )

    monkeypatch.setitem(configured_tracker._TRACKER_FACTORIES, "iou", RecordingTracker)
    tracker = create_tracker(GetCurrentConfig(CurrentConfig(Config())))

    tracker.reset()
    tracker.track_frame(detected_frame, count(1))
    tracker.reset()

    assert calls == ["reset", "frame", "reset"]


def test_every_valid_tracker_type_can_be_constructed() -> None:
    assert set(configured_tracker._TRACKER_FACTORIES) == set(TRACKER_TYPES)
