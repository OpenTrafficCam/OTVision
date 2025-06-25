from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from OTVision.dataformat import FRAME, INPUT_FILE_PATH, TRACK_ID
from OTVision.domain.detection import TrackId
from OTVision.domain.frame import (
    DetectedFrame,
    FinishedFrame,
    FrameNo,
    IsLastFrame,
    TrackedFrame,
)
from OTVision.track.model.filebased.frame_group import FrameGroup, get_output_file


@dataclass(frozen=True)
class FrameChunk:
    """
    A chunk of Frames obtained from a common file path source.

    Attributes:
        file (Path): common file path source of Frames.
        metadata (dict): otdet metadata.
        frames (Sequence[DetectedFrame]): a sequence of untracked Frames.
        frame_group_id (int): id of FrameGroup this FrameCHunk is part of.
    """

    file: Path
    metadata: dict
    frames: Sequence[DetectedFrame]
    frame_group_id: int

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
        frames (Sequence[TrackedFrame]): overrides frames
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
    frames: Sequence[TrackedFrame] = field(init=False)

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
        frames: Sequence[TrackedFrame],
        frame_group_id: int,
    ) -> None:

        object.__setattr__(self, "file", file)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "is_last_chunk", is_last_chunk)
        object.__setattr__(self, "frame_group_id", frame_group_id)

        observed = set().union(*(f.observed_tracks for f in frames))
        finished = set().union(
            *(f.finished_tracks for f in frames)
        )  # TODO remove discarded?
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
            frame_group_id=self.frame_group_id,
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"FG [{self.frame_group_id}] - {self.file}"


@dataclass(frozen=True)
class FinishedChunk(TrackedChunk):
    """A chunk of FinishedFrames.

    Attributes:
        frames (Sequence[FinishedFrame]): overrides frames
            with more specific frame type.
    """

    frames: Sequence[FinishedFrame]

    def to_detection_dicts(self) -> list[dict]:
        chunk_metadata = {INPUT_FILE_PATH: self.file.as_posix()}

        frames_progress = tqdm(
            self.frames, desc="Frames to_dict", total=len(self.frames), leave=False
        )

        detection_dict_list = [
            {**det_dict, **chunk_metadata}
            for frame in frames_progress
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
    def parse(
        self, file: Path, frame_group: FrameGroup, frame_offset: int
    ) -> FrameChunk:
        pass
