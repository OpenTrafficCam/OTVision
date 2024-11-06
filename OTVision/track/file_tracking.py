from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence, cast

from more_itertools import peekable

from OTVision.track.data import FinishedFrame, Frame, IsLastFrame, TrackedFrame, TrackId
from OTVision.track.tracking_interfaces import Tracker


@dataclass(frozen=True)
class FrameChunk:
    # start/end metadata?
    file: Path
    frames: Sequence[Frame[Path]]


@dataclass(frozen=True)
class TrackedChunk(FrameChunk):
    """
    is_last denotes this chunk is the last chunk,
    in that case the last frame is updates
    """

    is_last_chunk: bool
    frames: Sequence[TrackedFrame[Path]] = field(init=False)

    finished_tracks: set[TrackId] = field(init=False)
    observed_tracks: set[TrackId] = field(init=False)
    unfinished_tracks: set[TrackId] = field(init=False)
    last_track_frame: dict[TrackId, int] = field(init=False)

    def __init__(
        self, file: Path, is_last_chunk: bool, frames: Sequence[TrackedFrame[Path]]
    ) -> None:

        object.__setattr__(self, "file", file)
        object.__setattr__(self, "is_last_chunk", is_last_chunk)

        observed = set().union(*(f.observed_tracks for f in frames))
        finished = set().union(*(f.finished_tracks for f in frames))
        # TODO assert only observed tracks are finished?

        object.__setattr__(self, "finished_tracks", finished)
        object.__setattr__(self, "observed_tracks", observed)

        unfinished = {o for o in observed if o not in finished}

        if self.is_last_chunk:
            frames_list = list(frames)
            # assume frames sorted by occurrence

            last_frame = frames_list[-1]
            frames_list[-1] = replace(
                last_frame,
                # more trivial implementation: add all observed ids of entire chunk here
                # ->  all tracks are observed & finished in last frame or before
                # -> maybe large set with old ids that have already known to be finished
                finished_tracks=last_frame.finished_tracks.union(unfinished),
                unfinished_tracks=set(),
            )
            object.__setattr__(self, "frames", frames_list)

            unfinished = set()
        object.__setattr__(self, "unfinished_tracks", unfinished)

        # assume frames sorted by occurrence
        last_track_frame: dict[int, int] = {
            detection.track_id: frame.no
            for frame in self.frames
            for detection in frame.detections
        }
        object.__setattr__(self, "last_track_frame", last_track_frame)

    def finish(self, group_last_track_frame: dict[TrackId, int]) -> "FinishedChunk":
        is_last: IsLastFrame = (
            lambda frame_no, track_id: frame_no == group_last_track_frame[track_id]
        )
        return FinishedChunk(
            file=self.file,
            is_last_chunk=self.is_last_chunk,
            frames=[frame.finish(is_last) for frame in self.frames],
        )


@dataclass(frozen=True)
class FinishedChunk(TrackedChunk):
    frames: Sequence[FinishedFrame[Path]]


class ChunkParser(ABC):

    @abstractmethod
    def parser(self, file: Path) -> FrameChunk:
        pass


@dataclass(frozen=True)
class FrameGroup:
    files: list[Path]
    metadata: dict[Path, dict]
    # todo merge, update metadata


class FrameGroupParser(ABC):

    @abstractmethod
    def parse(self, file: Path) -> FrameGroup:
        pass

    @abstractmethod
    def merge(self, frameGroups: list[FrameGroup]) -> list[FrameGroup]:
        pass


class ChunkBasedTracker(Tracker[Path]):

    def __init__(self, tracker: Tracker[Path], chunkParser: ChunkParser) -> None:
        super().__init__()
        self._chunkParser = chunkParser
        self._tracker = tracker

    def track(self, frames: Iterable[Frame[Path]]) -> Iterable[TrackedFrame[Path]]:
        return self._tracker.track(frames)

    def track_chunk(self, chunk: FrameChunk, is_last_chunk: bool) -> TrackedChunk:
        tracked_frames = self.track(chunk.frames)
        return TrackedChunk(
            file=chunk.file, frames=list(tracked_frames), is_last_chunk=is_last_chunk
        )

    def track_file(self, file: Path, is_last_file: bool) -> TrackedChunk:
        chunk = self._chunkParser.parser(file)
        return self.track_chunk(chunk, is_last_file)


class GroupedFilesTracker(ChunkBasedTracker):

    def __init__(
        self,
        tracker: Tracker[Path],
        chunkParser: ChunkParser,
        frameGroupParser: FrameGroupParser,
    ) -> None:
        super().__init__(tracker, chunkParser)
        self._groupParser = frameGroupParser

    def track_group(self, group: FrameGroup) -> Iterable[TrackedChunk]:

        file_stream = peekable(group.files)
        for file in file_stream:
            is_last = file_stream.peek(default=None) is None
            yield self.track_file(file, is_last)

    def group_and_track_files(self, files: list[Path]) -> Iterable[TrackedChunk]:
        groups = [self._groupParser.parse(f) for f in files]
        merged = self._groupParser.merge(groups)

        for group in merged:
            yield from self.track_group(group)


class BufferedFinishedChunksTracker(GroupedFilesTracker):

    def __init__(
        self,
        tracker: Tracker[Path],
        chunkParser: ChunkParser,
        frameGroupParser: FrameGroupParser,
    ) -> None:
        super().__init__(tracker, chunkParser, frameGroupParser)
        self._unfinished_chunks: dict[TrackedChunk, set[TrackId]] = dict()
        self._merged_last_track_frame: dict[TrackId, int] = dict()

    def group_and_track_files(self, files: list[Path]) -> Iterable[FinishedChunk]:
        # reuse method but update type hint
        return cast(list[FinishedChunk], super().group_and_track_files(files))

    def track_group(self, group: FrameGroup) -> Iterable[FinishedChunk]:
        for chunk in super().track_group(group):

            self._merged_last_track_frame.update(chunk.last_track_frame)
            self._unfinished_chunks[chunk] = chunk.unfinished_tracks

            # update unfinished track ids of previously tracked chunks
            # if chunks have no pending tracks, make ready for finishing
            newly_finished_tracks = chunk.finished_tracks
            ready_chunks: list[TrackedChunk] = []
            for chunk, track_ids in self._unfinished_chunks.items():
                track_ids.difference_update(newly_finished_tracks)

                if len(track_ids) == 0:
                    ready_chunks.append(chunk)
                    del self._unfinished_chunks[chunk]

            finished_chunks = self._finish_chunks(ready_chunks)
            yield from finished_chunks

        # finish remaining chunks with pending tracks
        remaining_chunks = list(self._unfinished_chunks.keys())
        self._unfinished_chunks = dict()

        finished_chunks = self._finish_chunks(remaining_chunks)
        self._merged_last_track_frame = dict()
        yield from finished_chunks

    def _finish_chunks(self, chunks: list[TrackedChunk]) -> list[FinishedChunk]:
        # TODO sort finished chunks by start date?
        finished_chunks = [c.finish(self._merged_last_track_frame) for c in chunks]

        # the last frame of the observed tracks have been marked
        # track ids no longer required in _merged_last_track_frame
        finished_tracks = set().union(*(c.observed_tracks for c in chunks))
        self._merged_last_track_frame = {
            track_id: frame_no
            for track_id, frame_no in self._merged_last_track_frame.items()
            if track_id not in finished_tracks
        }

        return finished_chunks
