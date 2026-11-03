"""Tests for UnfinishedTracksBuffer.

Reproduces the KeyError that occurs when finishing the remaining containers
at the end of track_and_finish: a track id present in detections of a
still-unfinished container gets evicted from _merged_last_track_frame too
eagerly during a prior _finish_containers call.
"""

from dataclasses import dataclass, field
from typing import AsyncIterator

import pytest

from OTVision.domain.detection import TrackId
from OTVision.domain.frame import FrameNo, IsLastFrame
from OTVision.track.model.tracking_interfaces import UnfinishedTracksBuffer


@dataclass
class MockContainer:
    name: str
    last_frame: FrameNo
    # mapping of track_id -> last frame number where that track was observed
    # in this container (only for tracks with detections in this container).
    last_track_frame: dict[TrackId, FrameNo] = field(default_factory=dict)
    observed_tracks: set[TrackId] = field(default_factory=set)
    finished_tracks: set[TrackId] = field(default_factory=set)
    discarded_tracks: set[TrackId] = field(default_factory=set)

    @property
    def unfinished_tracks(self) -> set[TrackId]:
        return self.observed_tracks - self.finished_tracks - self.discarded_tracks


@dataclass
class FinishedMockContainer:
    name: str
    # tuples of (frame_no, track_id, is_last) recorded via the is_last callback
    is_last_calls: list[tuple[FrameNo, TrackId, bool]]


class MockBuffer(UnfinishedTracksBuffer[MockContainer, FinishedMockContainer]):
    def _get_last_track_frames(
        self, container: MockContainer
    ) -> dict[TrackId, FrameNo]:
        return container.last_track_frame

    def _get_unfinished_tracks(self, container: MockContainer) -> set[TrackId]:
        return set(container.unfinished_tracks)

    def _get_observed_tracks(self, container: MockContainer) -> set[TrackId]:
        return container.observed_tracks

    def _get_newly_finished_tracks(self, container: MockContainer) -> set[TrackId]:
        return container.finished_tracks

    def _get_newly_discarded_tracks(self, container: MockContainer) -> set[TrackId]:
        return container.discarded_tracks

    def _get_last_frame_of_container(self, container: MockContainer) -> FrameNo:
        return container.last_frame

    def _finish(
        self,
        container: MockContainer,
        is_last: IsLastFrame,
        discarded_tracks: set[TrackId],
        keep_discarded: bool,
    ) -> FinishedMockContainer:
        calls: list[tuple[FrameNo, TrackId, bool]] = []
        for track_id, frame_no in container.last_track_frame.items():
            result = is_last(frame_no, track_id)
            calls.append((frame_no, track_id, result))
        return FinishedMockContainer(name=container.name, is_last_calls=calls)


async def _astream(items: list[MockContainer]) -> AsyncIterator[MockContainer]:
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_keyerror_when_track_evicted_before_container_finishes() -> None:
    """Reproduces the production KeyError.

    Scenario:
        - Chunk A: observes tracks 544, 999 and X. None finished yet.
        - Chunk B: observes track 999 only; marks 544 as finished (the IOU
          tracker decided 544 is gone but did not observe 544 again).
        - Chunk C: observes a new track 777; marks 999 as finished. Track X
          (from A) is still unfinished.
        - Chunk D: marks X as finished.
        - Chunk E (last): marks 777 as finished, has no observations.

    Trace of the bug:
        When chunk C is processed, chunk B's only unfinished track (999) is
        finished, so chunk B becomes ready alone. _finish_containers([B])
        runs with last_frame_of_container = chunk B's last frame, and evicts
        track 544's entry (frame 10 <= chunk B's last frame). Chunk A is
        still unfinished because of track X. Later when X is finished, chunk
        A becomes ready and the finish call performs is_last(10, 544) ->
        KeyError because 544 was evicted.
    """
    chunk_a = MockContainer(
        name="A",
        last_frame=10,
        last_track_frame={544: 10, 999: 10, 111: 10},
        observed_tracks={544, 999, 111},
        finished_tracks=set(),
        discarded_tracks=set(),
    )
    chunk_b = MockContainer(
        name="B",
        last_frame=100,
        last_track_frame={999: 100},
        observed_tracks={999},
        finished_tracks={544},
        discarded_tracks=set(),
    )
    chunk_c = MockContainer(
        name="C",
        last_frame=200,
        last_track_frame={777: 200},
        observed_tracks={777},
        finished_tracks={999},
        discarded_tracks=set(),
    )
    chunk_d = MockContainer(
        name="D",
        last_frame=300,
        last_track_frame={},
        observed_tracks=set(),
        finished_tracks={111},
        discarded_tracks=set(),
    )
    chunk_e = MockContainer(
        name="E",
        last_frame=400,
        last_track_frame={},
        observed_tracks=set(),
        finished_tracks={777},
        discarded_tracks=set(),
    )

    buffer = MockBuffer(keep_discarded=True)

    finished: list[FinishedMockContainer] = []
    async for f in buffer.track_and_finish(
        _astream([chunk_a, chunk_b, chunk_c, chunk_d, chunk_e])
    ):
        finished.append(f)

    finished_by_name = {f.name: f for f in finished}
    assert set(finished_by_name.keys()) == {"A", "B", "C", "D", "E"}

    # chunk A's detection of 544 (frame 10) should be flagged is_last=True
    # because that is the last (and only) observation of 544.
    a_calls = {
        (tid, frame): is_last
        for frame, tid, is_last in finished_by_name["A"].is_last_calls
    }
    assert a_calls[(544, 10)] is True
    # chunk A's detection of 999 (frame 10) should be flagged is_last=False
    # because 999 was observed again at frame 100 in chunk B.
    assert a_calls[(999, 10)] is False
    # chunk A's detection of 111 (frame 10) — not observed elsewhere, so it is
    # the last frame for that track.
    assert a_calls[(111, 10)] is True
