from collections.abc import AsyncIterator
from datetime import datetime
from itertools import count

import pytest

from OTVision.domain.detection import Detection, TrackedDetection, TrackId
from OTVision.domain.frame import DetectedFrame, FinishedFrame, TrackedFrame
from OTVision.track.model.tracking_interfaces import (
    IdGenerator,
    Tracker,
    UnfinishedFramesBuffer,
)

_Step = tuple[list[TrackedDetection], set[TrackId], set[TrackId]]


class ScriptedTracker(Tracker):
    """Returns prepared detections, finished ids and discarded ids per frame."""

    def __init__(self, steps: list[_Step]) -> None:
        self._steps = iter(steps)

    def reset(self) -> None:
        pass

    def track_frame(
        self, frame: DetectedFrame, id_generator: IdGenerator
    ) -> TrackedFrame:
        detections, finished, discarded = next(self._steps)
        return TrackedFrame(
            no=frame.no,
            occurrence=frame.occurrence,
            source=frame.source,
            output=frame.output,
            detections=detections,
            finished_tracks=finished,
            discarded_tracks=discarded,
        )


async def _frame_group(frame_count: int) -> AsyncIterator[DetectedFrame]:
    for no in range(1, frame_count + 1):
        yield DetectedFrame(
            no=no,
            occurrence=datetime(2020, 1, 1),
            source="source",
            output="output",
            detections=[],
        )


async def _track(buffer: UnfinishedFramesBuffer, frame_count: int) -> list[int]:
    finished: list[FinishedFrame] = [
        frame async for frame in buffer.track(_frame_group(frame_count), count(1))
    ]
    return [d.track_id for frame in finished for d in frame.detections]


@pytest.mark.asyncio
async def test_discarded_track_does_not_drop_same_id_in_next_frame_group() -> None:
    car = Detection("car", 0.9, 50, 50, 20, 20)
    first_group: list[_Step] = [
        ([car.of_track(1, True)], set(), set()),
        # An empty frame is finished at once, while track 1 is still open.
        ([], set(), set()),
        ([], set(), {1}),
    ]
    # Track ids restart with every frame group.
    second_group: list[_Step] = [
        ([car.of_track(1, True)], set(), set()),
        ([], {1}, set()),
    ]
    buffer = UnfinishedFramesBuffer(
        ScriptedTracker(first_group + second_group), keep_discarded=False
    )

    first_ids = await _track(buffer, len(first_group))
    second_ids = await _track(buffer, len(second_group))

    assert first_ids == []
    assert second_ids == [1]
