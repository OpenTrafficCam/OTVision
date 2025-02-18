import logging
from pathlib import Path
from typing import Callable, Iterator

from more_itertools import peekable

from OTVision.config import CONFIG, DEFAULT_FILETYPE, OVERWRITE, TRACK
from OTVision.helpers.log import LOGGER_NAME
from OTVision.track.model.filebased.frame_chunk import (
    ChunkParser,
    FinishedChunk,
    FrameChunk,
    TrackedChunk,
)
from OTVision.track.model.filebased.frame_group import FrameGroup, FrameGroupParser
from OTVision.track.model.frame import (
    Frame,
    FrameNo,
    IsLastFrame,
    TrackedFrame,
    TrackId,
)
from OTVision.track.model.tracking_interfaces import (
    ID_GENERATOR,
    Tracker,
    UnfinishedTracksBuffer,
)

log = logging.getLogger(LOGGER_NAME)


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
