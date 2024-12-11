from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Generic, Iterator, Optional, Sequence, TypeVar

from PIL.Image import Image

TrackId = int


@dataclass(frozen=True, repr=True)
class Detection:
    label: str
    conf: float
    x: float
    y: float
    w: float
    h: float

    def of_track(self, id: TrackId, is_first: bool) -> "TrackedDetection":
        return TrackedDetection(
            self.label,
            self.conf,
            self.x,
            self.y,
            self.w,
            self.h,
            is_first,
            id,
        )


S = TypeVar("S")


@dataclass(frozen=True)
class Frame(Generic[S]):
    """
    source is generic, it can be the file path of the detection file
    or the url of a live data stream
    """

    no: int
    occurrence: datetime
    source: S
    detections: Sequence[Detection]
    image: Optional[Image]


@dataclass(frozen=True, repr=True)
class TrackedDetection(Detection):
    """
    is_first denotes this detection is the first occurrence of the assigned track id
    (
        the track id must be provided by the tracking algorithm,
        the tracking algorithm also knows whether the detection
        is the first occurrence of a track id,
        whether a detection is the last occurrence of a track id is
        not known to a frame based tracking algorithm,
        this information must be introduced by a some context aware wrapper
    )
    """

    is_first: bool
    track_id: TrackId

    def finish(self, is_last: bool) -> "FinishedDetection":
        return FinishedDetection.from_tracked_detection(self, is_last)

    def as_last_detection(self) -> "TrackedDetection":
        return FinishedDetection.from_tracked_detection(self, is_last=True)

    def as_intermediate_detection(self) -> "TrackedDetection":
        return FinishedDetection.from_tracked_detection(self, is_last=False)


@dataclass(frozen=True, repr=True)
class FinishedDetection(TrackedDetection):
    """
    is_last denotes this detection to be the last occurrence of the assigned track_id
    (
        whether a detection is the last occurrence of a track id is
        not known to a frame based tracking algorithm,
        this information must be introduced by a some context aware wrapper
    )
    """

    is_last: bool

    @classmethod
    def from_tracked_detection(
        cls, tracked_detection: TrackedDetection, is_last: bool
    ) -> "FinishedDetection":
        td = tracked_detection
        return cls(
            label=td.label,
            conf=td.conf,
            x=td.x,
            y=td.y,
            w=td.w,
            h=td.h,
            is_first=td.is_first,
            track_id=td.track_id,
            is_last=is_last,
        )


IsLastFrame = Callable[[int, TrackId], bool]


@dataclass(frozen=True)
class TrackedFrame(Frame[S]):
    """
    definitions

    detections - detections occurring in this frame, annotated with a track id
        (and whether it is the tracks first detection)
    observed_tracks - track ids that occur in this frame (derived value)
    finished_tracks - track ids of tracks observed in this or prior to this frame
        that can now be considered finished
        (these track ids should no longer be observed/assigned in future frames)
    """

    detections: Sequence[TrackedDetection]
    finished_tracks: set[TrackId]
    observed_tracks: set[TrackId] = field(init=False)
    unfinished_tracks: set[TrackId] = field(init=False)

    def __post_init__(self) -> None:
        observed = {d.track_id for d in self.detections}
        object.__setattr__(self, "observed_tracks", observed)

        unfinished = {o for o in self.observed_tracks if o not in self.finished_tracks}
        object.__setattr__(self, "unfinished_tracks", unfinished)

    def finish(self, is_last: IsLastFrame) -> "FinishedFrame":
        return FinishedFrame(
            no=self.no,
            occurrence=self.occurrence,
            source=self.source,
            finished_tracks=self.finished_tracks,
            detections=[
                det.finish(is_last(self.no, det.track_id)) for det in self.detections
            ],
            image=self.image,
        )


@dataclass(frozen=True)
class FinishedFrame(TrackedFrame[S]):
    """all detections are marked with is_last flag"""

    detections: Sequence[FinishedDetection]


C = TypeVar("C")  # Detection container
F = TypeVar("F")  # Finished container


class UnfinishedTracksBuffer(ABC, Generic[C, F]):

    def __init__(self) -> None:
        self._unfinished_containers: dict[C, set[TrackId]] = dict()
        self._merged_last_track_frame: dict[TrackId, int] = dict()

    @abstractmethod
    def _get_last_track_frames(self, container: C) -> dict[TrackId, int]:
        pass

    @abstractmethod
    def _get_unfinished_tracks(self, container: C) -> set[TrackId]:
        pass

    @abstractmethod
    def _get_observed_tracks(self, container: C) -> set[TrackId]:
        pass

    @abstractmethod
    def _get_newly_finished_tracks(self, container: C) -> set[TrackId]:
        pass

    @abstractmethod
    def _finish(self, container: C, is_last: IsLastFrame) -> F:
        pass

    def track_and_finish(self, containers: Iterator[C]) -> Iterator[F]:
        for container in containers:

            self._merged_last_track_frame.update(self._get_last_track_frames(container))
            self._unfinished_containers[container] = self._get_unfinished_tracks(
                container
            )

            # update unfinished track ids of previously tracked containers
            # if containers have no pending tracks, make ready for finishing
            newly_finished_tracks = self._get_newly_finished_tracks(container)
            ready_containers: list[C] = []
            for c, track_ids in self._unfinished_containers.items():
                track_ids.difference_update(newly_finished_tracks)

                if not track_ids:
                    ready_containers.append(c)
                    del self._unfinished_containers[c]

            finished_containers: list[F] = self._finish_containers(ready_containers)
            yield from finished_containers

        # finish remaining containers with pending tracks
        remaining_containers = list(self._unfinished_containers.keys())
        self._unfinished_containers = dict()

        finished_containers = self._finish_containers(remaining_containers)
        self._merged_last_track_frame = dict()
        yield from finished_containers

    def _finish_containers(self, containers: list[C]) -> list[F]:
        # TODO sort containers by occurrence / start date?
        is_last: IsLastFrame = (
            lambda frame_no, track_id: frame_no
            == self._merged_last_track_frame[track_id]
        )
        finished_containers: list[F] = [self._finish(c, is_last) for c in containers]

        # the last frame of the observed tracks have been marked
        # track ids no longer required in _merged_last_track_frame
        finished_tracks = set().union(
            *(self._get_observed_tracks(c) for c in containers)
        )
        self._merged_last_track_frame = {
            track_id: frame_no
            for track_id, frame_no in self._merged_last_track_frame.items()
            if track_id not in finished_tracks
        }

        return finished_containers
