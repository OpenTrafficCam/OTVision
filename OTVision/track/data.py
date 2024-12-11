from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Generic, Iterator, Optional, Sequence, TypeVar

from PIL.Image import Image

TrackId = int


@dataclass(frozen=True, repr=True)
class Detection:
    """Detection data without track context data.

    Attributes:
        label (str): Assigned label, e.g. vehicle class.
        conf (float): Confidence of detected class.
        x (float): X-coordinate of detection center.
        y (float): Y-coordinate of detection center.
        w (float): Width of detection.
        h (float): Height of detection.
    """

    label: str
    conf: float
    x: float
    y: float
    w: float
    h: float

    def of_track(self, id: TrackId, is_first: bool) -> "TrackedDetection":
        """Convert to TrackedDetection by adding track information.

        Args:
            id (TrackId): id of assigned track.
            is_first (bool): whether this detection is first of track.

        Returns:
            TrackedDetection: This detection data with additional track information.
        """
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
    """Frame metadata, optional image and respective detections.

    Attributes:
        no (int): Frame number.
        occurrence (datetime): Time stamp, at which frame was recorded.
        source (S): Generic source from where frame was obtained, e.g. video file path.
        detections (Sequence[Detection]): A sequence of Detections occurring in frame.
        image (Optional[Image]): Optional image data of frame.
    """

    no: int
    occurrence: datetime
    source: S
    detections: Sequence[Detection]
    image: Optional[Image]


@dataclass(frozen=True, repr=True)
class TrackedDetection(Detection):
    """Detection with additional track data.
    At the time a detection is tracked,
    it might not be known whether it is the last of a track.

    Attributes:
        is_first (bool): whether this detection is the first in the track.
        track_id (TrackId): id of the assigned track.
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
    """Detection data with extended track information including is_finished.

    Attributes:
        is_last (bool): whether this detection is the last in the track.
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
    """Frame metadata with tracked detections.
    Also provides additional aggregated information about:
    observed, finished and unfinished tracks.

    Attributes:
        detections (Sequence[TrackedDetection]): overrides Frame.detections with more
            specific type of detection.
        observed_tracks (set[TrackId]): set of tracks of which detection occur in this
            frame.
        finished_tracks (set[TrackId]): track ids of tracks observed in this or prior
            to this frame that can now be considered finished. These track ids should
            no longer be observed/assigned in future frames.
        unfinished_tracks (set[TrackId]): ob served tracks that are not yet finished.
    """

    detections: Sequence[TrackedDetection]
    finished_tracks: set[TrackId]
    observed_tracks: set[TrackId] = field(init=False)
    unfinished_tracks: set[TrackId] = field(init=False)

    def __post_init__(self) -> None:
        # derive observed and unfinished tracks
        # from tracked detections and finished track information
        observed = {d.track_id for d in self.detections}
        object.__setattr__(self, "observed_tracks", observed)

        unfinished = {o for o in self.observed_tracks if o not in self.finished_tracks}
        object.__setattr__(self, "unfinished_tracks", unfinished)

    def finish(self, is_last: IsLastFrame) -> "FinishedFrame":
        """Turn this TrackedFrame into a finished frame
        by adding is_finished information to all its detection.

        Args:
            is_last (IsLastFrame): function to determine whether
                a track is finished in a certain frame.

        Returns:
            FinishedFrame: frame with FinishedDetections
        """
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
    """TrackedFrame with FinishedDetections.

    Args:
        detections (Sequence[FinishedDetection]): overrides TrackedFrame.detections
            with more specific detection type.
    """

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
