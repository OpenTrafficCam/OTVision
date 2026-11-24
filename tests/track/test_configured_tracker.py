from dataclasses import replace
from datetime import datetime
from itertools import count

import pytest

from OTVision.application.config import Config, TrackConfig
from OTVision.application.config_parser import InvalidOtvisionConfigError
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.domain.current_config import CurrentConfig
from OTVision.domain.detection import Detection
from OTVision.domain.frame import DetectedFrame
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
