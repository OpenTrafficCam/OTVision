import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Sequence

from more_itertools import peekable

from OTVision.config import CONFIG, DEFAULT_FILETYPE, OVERWRITE, TRACK
from OTVision.dataformat import FRAME, INPUT_FILE_PATH, TRACK_ID
from OTVision.helpers.log import LOGGER_NAME
from OTVision.track.data import (
    FinishedFrame,
    Frame,
    FrameNo,
    IsLastFrame,
    TrackedFrame,
    TrackId,
)
from OTVision.track.tracking_interfaces import (
    ID_GENERATOR,
    Tracker,
    UnfinishedTracksBuffer,
)

log = logging.getLogger(LOGGER_NAME)


def get_output_file(file: Path, with_suffix: str) -> Path:
    return file.with_suffix(with_suffix)


@dataclass(frozen=True)
class FrameChunk:
    """
    A chunk of Frames obtained from a common file path source.

    Attributes:
        file (Path): common file path source of Frames.
        metadata (dict): otdet metadata.
        frames (Sequence[Frame[Path]]): a sequence of untracked Frames.
    """

    file: Path
    metadata: dict
    frames: Sequence[Frame[Path]]

    def check_output_file_exists(self, with_suffix: str) -> bool:
        return get_output_file(self.file, with_suffix).is_file()


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
    discarded_tracks: set[TrackId] = field(init=False)
    last_track_frame: dict[TrackId, int] = field(init=False)

    def __init__(
        self,
        file: Path,
        metadata: dict,
        is_last_chunk: bool,
        frames: Sequence[TrackedFrame[Path]],
    ) -> None:

        object.__setattr__(self, "file", file)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "is_last_chunk", is_last_chunk)

        observed = set().union(*(f.observed_tracks for f in frames))
        finished = set().union(*(f.finished_tracks for f in frames))
        discarded = set().union(*(f.discarded_tracks for f in frames))
        unfinished = {o for o in observed if o not in finished and o not in discarded}

        # set all unfinished tracks as finished, as this is the last track
        if self.is_last_chunk:
            frames_list = list(frames)
            # assume frames sorted by occurrence

            last_frame = frames_list[-1]
            frames_list[-1] = replace(
                last_frame,
                finished_tracks=last_frame.finished_tracks.union(unfinished),
            )

            unfinished = set()
            finished = set().union(*(f.finished_tracks for f in frames_list))
        else:
            frames_list = list(frames)

        object.__setattr__(self, "frames", frames_list)

        object.__setattr__(self, "finished_tracks", finished)
        object.__setattr__(self, "observed_tracks", observed)
        object.__setattr__(self, "unfinished_tracks", unfinished)
        object.__setattr__(self, "discarded_tracks", discarded)

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
            metadata=self.metadata,
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

    def to_detection_dicts(self) -> list[dict]:
        chunk_metadata = {INPUT_FILE_PATH: self.file.as_posix()}  # TODO posix here?

        detection_dict_list = [
            {**det_dict, **chunk_metadata}
            for frame in self.frames
            for det_dict in frame.to_detection_dicts()
        ]

        detection_dict_list.sort(
            key=lambda detection: (
                detection[FRAME],
                detection[TRACK_ID],
            )
        )
        return detection_dict_list


class ChunkParser(ABC):
    """A parser for file path to FrameChunk."""

    @abstractmethod
    def parse(self, file: Path, frame_offset: int) -> FrameChunk:
        pass


@dataclass(frozen=True)
class FrameGroup:
    id: int
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
            id=first.id,
            start_date=first.start_date,
            end_date=second.end_date,
            hostname=first.hostname,
            files=files,
            metadata_by_file=metadata,
        )

        return merged

    def check_any_output_file_exists(self, with_suffix: str) -> bool:
        return len(self.get_existing_output_files(with_suffix)) > 0

    def get_existing_output_files(self, with_suffix: str) -> list[Path]:
        return [file for file in self.get_output_files(with_suffix) if file.is_file()]

    def get_output_files(self, with_suffix: str) -> list[Path]:
        return [get_output_file(file, with_suffix) for file in self.files]

    def with_id(self, new_id: int) -> "FrameGroup":
        return replace(self, id=new_id)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"FrameGroup[{self.id}] = [{self.start_date} - {self.end_date}]"


class FrameGroupParser(ABC):

    def process_all(self, files: list[Path]) -> list[FrameGroup]:
        parsed: list[FrameGroup] = [self.parse(file) for file in files]
        merged: list[FrameGroup] = self.merge(parsed)
        updated: list[FrameGroup] = [
            self.updated_metadata_copy(group).with_id(i)
            for i, group in enumerate(merged)
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
            file=chunk.file,
            frames=list(tracked_frames),
            metadata=chunk.metadata,
            is_last_chunk=is_last_chunk,
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
        overwrite: bool = CONFIG[TRACK][OVERWRITE],
        file_type: str = CONFIG[DEFAULT_FILETYPE][TRACK],
    ) -> None:
        super().__init__(tracker, chunk_parser)
        self._group_parser = frame_group_parser
        self._id_generator_of = id_generator_factory
        self._overwrite = overwrite
        self._file_type = file_type

    def track_group(self, group: FrameGroup) -> Iterator[TrackedChunk]:
        if self.check_skip_due_to_existing_output_files(group):
            yield from []  # TODO how to create empty generator stream?

        frame_offset = 0  # frame no starts a 0 for each frame group
        id_generator = self._id_generator_of(group)  # new id generator per group
        file_stream = peekable(group.files)
        for file in file_stream:
            is_last = file_stream.peek(default=None) is None

            chunk = self._chunk_parser.parse(file, frame_offset)
            frame_offset = chunk.frames[-1].no + 1  # assuming frames are sorted by no

            tracked_chunk = self.track_chunk(chunk, is_last, id_generator)
            yield tracked_chunk

    def group_and_track_files(self, files: list[Path]) -> Iterator[TrackedChunk]:
        processed = self._group_parser.process_all(files)

        for group in processed:
            yield from self.track_group(group)

    def check_skip_due_to_existing_output_files(self, group: FrameGroup) -> bool:
        if not self._overwrite and group.check_any_output_file_exists(self._file_type):
            existing_files = group.get_existing_output_files(
                with_suffix=self._file_type
            )
            log.warning(
                (
                    f"{existing_files} already exist(s)."
                    "To overwrite, set overwrite to True"
                )
            )
            return True

        return False


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
