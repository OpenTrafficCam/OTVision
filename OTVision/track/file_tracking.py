from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Sequence, cast

from more_itertools import peekable

from OTVision.track.data import (
    FinishedFrame,
    Frame,
    FrameNo,
    IsLastFrame,
    TrackedFrame,
    TrackId,
    UnfinishedTracksBuffer,
)
from OTVision.track.tracking_interfaces import ID_GENERATOR, Tracker


@dataclass(frozen=True)
class FrameChunk:
    """
    A chunk of Frames obtained from a common file path source.

    Attributes:
        file (Path): common file path source of Frames.
        frames (Sequence[Frame[Path]]): a sequence of untracked Frames.
    """

    # TODO start/end metadata?
    file: Path
    frames: Sequence[Frame[Path]]


@dataclass(frozen=True)
class TrackedChunk(FrameChunk):
    """A chunk of TrackedFrames extends FrameChunk.
    Aggregates observed/finished/unfinished tracking metadata.

    If is_last_chunk is true, all unfinished tracks
    are marked as finished in last frame.

    Attributes:
        is_last_chunk (bool): whether this chunk is the last of
            subsequently (related/connected) chunks.
        frames (Sequence[TrackedFrame[Path]]): overrides frames
            with more specific frame type.
        finished_tracks (set[TrackId]): aggregates finished tracks
            of given TrackedFrames.
        observed_tracks (set[TrackId]): aggregates observed tracks
            of given TrackedFrames.
        unfinished_tracks (set[TrackId]): aggregates unfinished tracks
            of given TrackedFrames as observed but not finished tracks in chunk.
        last_track_frame (dict[TrackId, int]): mapping of track id
            to frame number in which it last occurs.
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
        unfinished = {o for o in observed if o not in finished}

        # set all unfinished tracks as finished, as this is the last track
        if self.is_last_chunk:
            frames_list = list(frames)
            # assume frames sorted by occurrence

            last_frame = frames_list[-1]
            frames_list[-1] = replace(
                last_frame,
                # more trivial implementation? add all observed ids of entire chunk here
                # ->  all tracks are observed & finished in last frame or before
                # -> maybe large set with old ids that have already known to be finished
                finished_tracks=last_frame.finished_tracks.union(unfinished),
                unfinished_tracks=set(),
            )
            object.__setattr__(self, "frames", frames_list)

            unfinished = set()
            finished = set().union(*(f.finished_tracks for f in frames_list))

        object.__setattr__(self, "finished_tracks", finished)
        object.__setattr__(self, "observed_tracks", observed)
        object.__setattr__(self, "unfinished_tracks", unfinished)

        # assume frames sorted by occurrence
        last_track_frame: dict[TrackId, FrameNo] = {
            detection.track_id: frame.no
            for frame in self.frames
            for detection in frame.detections
        }
        object.__setattr__(self, "last_track_frame", last_track_frame)

    def finish(
        self,
        is_last: IsLastFrame,
        discarded_tracks: set[TrackId],
        keep_discarded: bool = False,
    ) -> "FinishedChunk":
        """Turn this TrackedChunk into a FinishedChunk
        by adding is_finished information to all its detections.

        Args:
            is_last (IsLastFrame): function to determine whether
                a track is finished in a certain frame.
            discarded_tracks (set[TrackId]): list of tracks considered discarded.
                Used to mark corresponding tracks.
            keep_discarded (bool): whether FinishedDetections marked as discarded
                should be kept in detections list. Defaults to False.

        Returns:
            FinishedChunk: chunk of FinishedFrames
        """
        return FinishedChunk(
            file=self.file,
            is_last_chunk=self.is_last_chunk,
            frames=[
                frame.finish(is_last, discarded_tracks, keep_discarded)
                for frame in self.frames
            ],
        )


@dataclass(frozen=True)
class FinishedChunk(TrackedChunk):
    """A chunk of FinishedFrames.

    Attributes:
        frames (Sequence[FinishedFrame[Path]]): overrides frames
            with more specific frame type.
    """

    frames: Sequence[FinishedFrame[Path]]


class ChunkParser(ABC):
    """A parser for file path to FrameChunk."""

    @abstractmethod
    def parse(self, file: Path, frame_offset: int) -> FrameChunk:
        pass


@dataclass(frozen=True)
class FrameGroup:
    start_date: datetime
    end_date: datetime
    hostname: str
    files: list[Path]
    metadata_by_file: dict[Path, dict]  # TODO originally key is posix, why?

    def merge(self, other: "FrameGroup") -> "FrameGroup":
        if self.start_date < other.start_date:
            return FrameGroup._merge(self, other)
        else:
            return FrameGroup._merge(other, self)

    @staticmethod
    def _merge(first: "FrameGroup", second: "FrameGroup") -> "FrameGroup":
        if first.hostname != second.hostname:
            raise ValueError("Hostname of FrameGroups does not match")

        files = first.files + second.files
        metadata = dict(first.metadata_by_file)
        metadata.update(second.metadata_by_file)

        merged = FrameGroup(
            start_date=first.start_date,
            end_date=second.end_date,
            hostname=first.hostname,
            files=files,
            metadata_by_file=metadata,
        )

        return merged

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"{self.start_date} - {self.end_date}"


class FrameGroupParser(ABC):

    def process_all(self, files: list[Path]) -> list[FrameGroup]:
        parsed: list[FrameGroup] = [self.parse(file) for file in files]
        merged: list[FrameGroup] = self.merge(parsed)
        updated: list[FrameGroup] = [
            self.updated_metadata_copy(group) for group in merged
        ]

        return updated

    @abstractmethod
    def parse(self, file: Path) -> FrameGroup:
        pass

    @abstractmethod
    def merge(self, frame_groups: list[FrameGroup]) -> list[FrameGroup]:
        pass

    def updated_metadata_copy(self, frame_group: FrameGroup) -> FrameGroup:
        new_metadata = self.update_metadata(frame_group)
        return replace(frame_group, metadata_by_file=new_metadata)

    @abstractmethod
    def update_metadata(self, frame_group: FrameGroup) -> dict[Path, dict]:
        pass


class ChunkBasedTracker(Tracker[Path]):

    def __init__(self, tracker: Tracker[Path], chunkParser: ChunkParser) -> None:
        super().__init__()
        self._chunk_parser = chunkParser
        self._tracker = tracker

    def track_frame(
        self,
        frames: Frame[Path],
        id_generator: ID_GENERATOR,
    ) -> TrackedFrame[Path]:
        return self._tracker.track_frame(frames, id_generator)

    def track_chunk(
        self,
        chunk: FrameChunk,
        is_last_chunk: bool,
        id_generator: ID_GENERATOR,
    ) -> TrackedChunk:
        tracked_frames = self.track(iter(chunk.frames), id_generator)
        return TrackedChunk(
            file=chunk.file, frames=list(tracked_frames), is_last_chunk=is_last_chunk
        )

    def track_file(
        self,
        file: Path,
        is_last_file: bool,
        id_generator: ID_GENERATOR,
        frame_offset: int = 0,
    ) -> TrackedChunk:
        chunk = self._chunk_parser.parse(file, frame_offset)
        return self.track_chunk(chunk, is_last_file, id_generator)


ID_GENERATOR_FACTORY = Callable[[FrameGroup], ID_GENERATOR]


class GroupedFilesTracker(ChunkBasedTracker):

    def __init__(
        self,
        tracker: Tracker[Path],
        chunk_parser: ChunkParser,
        frame_group_parser: FrameGroupParser,
        id_generator_factory: ID_GENERATOR_FACTORY,
    ) -> None:
        super().__init__(tracker, chunk_parser)
        self._group_parser = frame_group_parser
        self._id_generator_of = id_generator_factory

    def track_group(self, group: FrameGroup) -> Iterator[TrackedChunk]:
        frame_offset = 0  # frame no starts a 0 for each frame group
        id_generator = self._id_generator_of(group)  # new id generator per group
        file_stream = peekable(group.files)
        for file in file_stream:
            is_last = file_stream.peek(default=None) is None
            chunk = self.track_file(file, is_last, id_generator, frame_offset)
            frame_offset = chunk.frames[-1].no + 1  # assuming frames are sorted by no
            yield chunk

    def group_and_track_files(self, files: list[Path]) -> Iterator[TrackedChunk]:
        processed = self._group_parser.process_all(files)

        for group in processed:
            yield from self.track_group(group)


class UnfinishedChunksBuffer(UnfinishedTracksBuffer[TrackedChunk, FinishedChunk]):

    def __init__(
        self,
        tracker: GroupedFilesTracker,
        keep_discarded: bool = False,
    ) -> None:
        super().__init__(keep_discarded)
        self.tracker = tracker

    def track_group(self, group: FrameGroup) -> Iterator[FinishedChunk]:
        tracked_chunk_stream = self.tracker.track_group(group)
        return self.track_and_finish(tracked_chunk_stream)

    def _get_last_track_frames(self, container: TrackedChunk) -> dict[TrackId, FrameNo]:
        return container.last_track_frame

    def _get_unfinished_tracks(self, container: TrackedChunk) -> set[TrackId]:
        return container.unfinished_tracks

    def _get_observed_tracks(self, container: TrackedChunk) -> set[TrackId]:
        return container.observed_tracks

    def _get_newly_finished_tracks(self, container: TrackedChunk) -> set[TrackId]:
        return container.finished_tracks

    def _finish(
        self,
        container: TrackedChunk,
        is_last: IsLastFrame,
        discarded_tracks: set[TrackId],
        keep_discarded: bool,
    ) -> FinishedChunk:
        return container.finish(is_last, discarded_tracks, keep_discarded)


class OldBufferedFinishedChunksTracker(GroupedFilesTracker):

    def __init__(
        self,
        tracker: Tracker[Path],
        chunk_parser: ChunkParser,
        frame_group_parser: FrameGroupParser,
        id_generator_factory: ID_GENERATOR_FACTORY,
    ) -> None:
        super().__init__(
            tracker, chunk_parser, frame_group_parser, id_generator_factory
        )
        self._unfinished_chunks: dict[TrackedChunk, set[TrackId]] = dict()
        self._merged_last_track_frame: dict[TrackId, FrameNo] = dict()

    def group_and_track_files(self, files: list[Path]) -> Iterator[FinishedChunk]:
        # reuse method but update type hint
        return cast(Iterator[FinishedChunk], super().group_and_track_files(files))

    def track_group(self, group: FrameGroup) -> Iterator[FinishedChunk]:
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
        is_last: IsLastFrame = (
            lambda frame_no, track_id: frame_no
            == self._merged_last_track_frame[track_id]
        )
        finished_chunks = [c.finish(is_last, set()) for c in chunks]

        # the last frame of the observed tracks have been marked
        # track ids no longer required in _merged_last_track_frame
        finished_tracks = set().union(*(c.observed_tracks for c in chunks))
        self._merged_last_track_frame = {
            track_id: frame_no
            for track_id, frame_no in self._merged_last_track_frame.items()
            if track_id not in finished_tracks
        }

        return finished_chunks
