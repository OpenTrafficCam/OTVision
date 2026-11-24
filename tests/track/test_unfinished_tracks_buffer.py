from datetime import timedelta
from pathlib import Path
from typing import AsyncIterator, TypeVar
from unittest.mock import Mock

import pytest

from OTVision.domain.detection import TrackedDetection
from OTVision.domain.frame import FinishedFrame, TrackedFrame
from OTVision.track.model.filebased.frame_chunk import FinishedChunk, TrackedChunk
from OTVision.track.model.tracking_interfaces import Tracker, UnfinishedFramesBuffer
from OTVision.track.tracker.filebased_tracking import (
    GroupedFilesTracker,
    UnfinishedChunksBuffer,
)
from tests.track.helper.data_builder import DEFAULT_START_DATE

T = TypeVar("T")


async def _stream(*containers: T) -> AsyncIterator[T]:
    for container in containers:
        yield container


def _tracked_frame(
    no: int,
    *tracks: tuple[int, bool],
    finished: tuple[int, ...] = (),
    discarded: tuple[int, ...] = (),
) -> TrackedFrame:
    return TrackedFrame(
        no=no,
        occurrence=DEFAULT_START_DATE + timedelta(seconds=no),
        source=f"clip-{no}.otdet",
        output=f"clip-{no}.ottrk",
        detections=[
            TrackedDetection(
                label="car",
                conf=0.9,
                x=track_id,
                y=5,
                w=5,
                h=5,
                is_first=is_first,
                track_id=track_id,
            )
            for track_id, is_first in tracks
        ],
        finished_tracks=set(finished),
        discarded_tracks=set(discarded),
    )


def _tracked_chunk(
    frame: TrackedFrame, *, is_last_chunk: bool, frame_group_id: int = 1
) -> TrackedChunk:
    return TrackedChunk(
        file=Path(frame.source),
        metadata={},
        is_last_chunk=is_last_chunk,
        frames=[frame],
        frame_group_id=frame_group_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("keep_discarded", [False, True])
@pytest.mark.parametrize("as_chunks", [False, True], ids=["frames", "chunks"])
async def test_finishes_pending_containers_after_later_ready_container(
    keep_discarded: bool, as_chunks: bool
) -> None:
    first = _tracked_frame(1, (1, True), (2, True), (3, True))
    second = _tracked_frame(2, (1, False), (2, False))
    # Track 1 is still pending at EOF. Track 2 was discarded, and track 3
    # finishes in the short final container after also appearing in the first.
    last = _tracked_frame(3, (3, False), (4, True), finished=(3, 4), discarded=(2,))

    buffer: UnfinishedChunksBuffer | UnfinishedFramesBuffer
    results: AsyncIterator[FinishedChunk | FinishedFrame]
    if as_chunks:
        buffer = UnfinishedChunksBuffer(Mock(spec=GroupedFilesTracker), keep_discarded)
        chunks = [
            _tracked_chunk(frame, is_last_chunk=frame is last)
            for frame in (first, second, last)
        ]
        results = buffer.track_and_finish(_stream(*chunks))
    else:
        buffer = UnfinishedFramesBuffer(Mock(spec=Tracker), keep_discarded)
        results = buffer.track_and_finish(_stream(first, second, last))

    ready = await anext(results)
    ready_frames = ready.frames if isinstance(ready, FinishedChunk) else [ready]
    assert [frame.no for frame in ready_frames] == [3]
    assert ready_frames[0].detections == [
        last.detections[0].as_last_detection(is_discarded=False),
        last.detections[1].as_last_detection(is_discarded=False),
    ]
    last_frames_after_ready = buffer._merged_last_track_frame.copy()

    pending = [
        frame
        async for result in results
        for frame in (result.frames if isinstance(result, FinishedChunk) else [result])
    ]
    assert [frame.no for frame in pending] == [1, 2]
    expected_first = [first.detections[0].as_intermediate_detection(is_discarded=False)]
    expected_second = [second.detections[0].as_last_detection(is_discarded=False)]
    if keep_discarded:
        expected_first.append(
            first.detections[1].as_intermediate_detection(is_discarded=True)
        )
        expected_second.append(
            second.detections[1].as_last_detection(is_discarded=True)
        )
    expected_first.append(
        first.detections[2].as_intermediate_detection(is_discarded=False)
    )
    assert pending[0].detections == expected_first
    assert pending[1].detections == expected_second
    assert last_frames_after_ready == {1: 2, 2: 2, 3: 3}
    assert not buffer._unfinished_containers
    assert not buffer._merged_last_track_frame
    assert not buffer._discarded_tracks


@pytest.mark.asyncio
@pytest.mark.parametrize("keep_discarded", [False, True])
async def test_discarded_track_id_can_be_reused_in_next_group(
    keep_discarded: bool,
) -> None:
    buffer = UnfinishedChunksBuffer(Mock(spec=GroupedFilesTracker), keep_discarded)
    # The discard event has no matching detection in this group.
    empty = _tracked_frame(1, discarded=(1,))
    first_group = [
        chunk
        async for chunk in buffer.track_and_finish(
            _stream(_tracked_chunk(empty, is_last_chunk=True))
        )
    ]
    assert len(first_group) == 1
    assert first_group[0].frames[0].detections == []

    discarded_after_first_group = buffer._discarded_tracks.copy()
    restarted = _tracked_frame(1, (1, True))
    next_group = [
        chunk
        async for chunk in buffer.track_and_finish(
            _stream(_tracked_chunk(restarted, is_last_chunk=True, frame_group_id=2))
        )
    ]
    assert len(next_group) == 1
    assert next_group[0].frames[0].detections == [
        restarted.detections[0].as_last_detection(is_discarded=False)
    ]
    assert not discarded_after_first_group
    assert not buffer._discarded_tracks


@pytest.mark.asyncio
async def test_aborted_group_does_not_leak_state_into_next_group() -> None:
    buffer = UnfinishedChunksBuffer(Mock(spec=GroupedFilesTracker), False)
    # Track 2 keeps the chunk pending while track 1 is marked as discarded.
    pending = _tracked_frame(1, (2, True), discarded=(1,))

    async def aborted_group() -> AsyncIterator[TrackedChunk]:
        yield _tracked_chunk(pending, is_last_chunk=False)
        raise RuntimeError("reading the next file failed")

    with pytest.raises(RuntimeError):
        _ = [chunk async for chunk in buffer.track_and_finish(aborted_group())]

    restarted = _tracked_frame(1, (1, True))
    next_group = [
        chunk
        async for chunk in buffer.track_and_finish(
            _stream(_tracked_chunk(restarted, is_last_chunk=True, frame_group_id=2))
        )
    ]
    assert [chunk.frames[0].no for chunk in next_group] == [1]
    assert next_group[0].frames[0].detections == [
        restarted.detections[0].as_last_detection(is_discarded=False)
    ]
