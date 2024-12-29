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

    def finish(self, is_last: bool, is_discarded: bool) -> "FinishedDetection":
        return FinishedDetection.from_tracked_detection(self, is_last, is_discarded)

    def as_last_detection(self, is_discarded: bool) -> "TrackedDetection":
        return FinishedDetection.from_tracked_detection(
            self, is_last=True, is_discarded=is_discarded
        )

    def as_intermediate_detection(self, is_discarded: bool) -> "TrackedDetection":
        return FinishedDetection.from_tracked_detection(
            self, is_last=False, is_discarded=is_discarded
        )


@dataclass(frozen=True, repr=True)
class FinishedDetection(TrackedDetection):
    """Detection data with extended track information including is_finished.

    Attributes:
        is_last (bool): whether this detection is the last in the track.
        is_discarded (bool): whether the detections's track was discarded.
    """

    is_last: bool
    is_discarded: bool

    @classmethod
    def from_tracked_detection(
        cls, tracked_detection: TrackedDetection, is_last: bool, is_discarded: bool
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
            is_discarded=is_discarded,
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
            no longer be observed/assigned in future frames. (successfully completed)
        discarded_tracks (set[TrackId]): track ids, that are now considered discarded.
            The corresponding tracks are no longer pursued, previous TrackedDetections
            of these tracks are also considered discarded. Discarded tracks may be
            observed but not finished.(unsuccessful, incomplete)
        unfinished_tracks (set[TrackId]): observed tracks that are not yet finished
            and were not discarded.
    """

    detections: Sequence[TrackedDetection]
    finished_tracks: set[TrackId]
    discarded_tracks: set[TrackId]
    observed_tracks: set[TrackId] = field(init=False)
    unfinished_tracks: set[TrackId] = field(init=False)

    def __post_init__(self) -> None:
        # derive observed and unfinished tracks
        # from tracked detections and finished track information
        observed = {d.track_id for d in self.detections}
        object.__setattr__(self, "observed_tracks", observed)

        unfinished = {
            o
            for o in self.observed_tracks
            if o not in self.finished_tracks and o not in self.discarded_tracks
        }
        object.__setattr__(self, "unfinished_tracks", unfinished)

    def finish(
        self,
        is_last: IsLastFrame,
        discarded_tracks: set[TrackId],
        keep_discarded: bool = False,
    ) -> "FinishedFrame":
        """Turn this TrackedFrame into a finished frame
        by adding is_finished information to all its detections.

        Args:
            is_last (IsLastFrame): function to determine whether
                a track is finished in a certain frame.
            discarded_tracks (set[TrackId]): list of tracks considered discarded.
                Used to mark corresponding tracks.
            keep_discarded (bool): whether FinishedDetections marked as discarded
                should be kept in detections list. Defaults to False.
        Returns:
            FinishedFrame: frame with FinishedDetections
        """
        if keep_discarded:
            detections = [
                det.finish(
                    is_last=is_last(self.no, det.track_id),
                    is_discarded=(det.track_id in discarded_tracks),
                )
                for det in self.detections
            ]
        else:
            detections = [
                det.finish(is_last=is_last(self.no, det.track_id), is_discarded=False)
                for det in self.detections
                if (det.track_id not in discarded_tracks)
            ]

        return FinishedFrame(
            no=self.no,
            occurrence=self.occurrence,
            source=self.source,
            finished_tracks=self.finished_tracks,
            detections=detections,
            image=self.image,
            discarded_tracks=discarded_tracks,
        )


@dataclass(frozen=True)
class FinishedFrame(TrackedFrame[S]):
    """TrackedFrame with FinishedDetections.

    Args:
        detections (Sequence[FinishedDetection]): overrides TrackedFrame.detections
            with more specific detection type.
    """

    detections: Sequence[FinishedDetection]


C = TypeVar("C")  # Detection container: e.g. TrackedFrame or TrackedChunk
F = TypeVar("F")  # Finished container: e.g. FinishedFrame or FinishedChunk


class UnfinishedTracksBuffer(ABC, Generic[C, F]):
    """UnfinishedTracksBuffer provides functionality
    to add finished information to tracked detections.

    It processes containers (C) of TrackedDetections, buffers them
    and stores track ids that are reported as finished.
    Only when all tracks of a container (C) were marked as finished,
    it is converted into a finished container (F) and yielded.

    Args:
        Generic (C): generic type of TrackedDetection container
            (e.g. TrackedFrame or TrackedChunk)
        Generic (F): generic type of FinishedDetection container
            (e.g. FinishedFrame or FinishedChunk)
        keep_discarded (bool): whether detections marked as discarded should
            be kept of filtered when finishing them. Defaults to False.
    """

    def __init__(self, keep_discarded: bool = False) -> None:
        self._keep_discarded = keep_discarded
        self._unfinished_containers: dict[C, set[TrackId]] = dict()
        self._merged_last_track_frame: dict[TrackId, int] = dict()
        self._discarded_tracks: set[TrackId] = set()

    @abstractmethod
    def _get_last_track_frames(self, container: C) -> dict[TrackId, int]:
        """Mapping from TrackId to frame no of last detection occurrence.
        Mapping for all tracks in newly tracked container.

        Args:
            container (C): newly tracked TrackedDetection container

        Returns:
            dict[TrackId, int]: last frame no by TrackId
        """
        pass

    @abstractmethod
    def _get_unfinished_tracks(self, container: C) -> set[TrackId]:
        """TrackIds of given container, that are marked as unfinished.

        Args:
            container (C): newly tracked TrackedDetection container

        Returns:
            set[TrackId]: TrackIds of container marked as unfinished
        """
        pass

    @abstractmethod
    def _get_observed_tracks(self, container: C) -> set[TrackId]:
        """TrackIds observed given (newly tracked) container.

        Args:
            container (C): newly tracked TrackedDetection container

        Returns:
            set[TrackId]: observed TrackIds of container
        """
        pass

    @abstractmethod
    def _get_newly_finished_tracks(self, container: C) -> set[TrackId]:
        """TrackIds marked as finished in the given (newly tracked) container.

        Args:
            container (C): newly tracked TrackedDetection container

        Returns:
            set[TrackId]: finished TrackIds in container
        """
        pass

    @abstractmethod
    def _get_newly_discarded_tracks(self, container: C) -> set[TrackId]:
        """TrackIds marked as discarded in the given (newly tracked) container.

        Args:
            container (C): newly tracked TrackedDetection container

        Returns:
            set[TrackId]: discarded TrackIds in container
        """
        pass

    @abstractmethod
    def _finish(
        self,
        container: C,
        is_last: IsLastFrame,
        discarded_tracks: set[TrackId],
        keep_discarded: bool,
    ) -> F:
        """Transform the given container to a finished container
        by adding is_finished information to all contained TrackedDetections
        turning them into FinishedDetections.

        Args:
            container (C): container of TrackedDetections
            is_last (IsLastFrame): check whether a track ends in a certain frame
            keep_discarded (bool): whether detections marked as discarded are kept.
        Returns:
            F: a finished container with transformed detections of given container
        """
        pass

    def track_and_finish(self, containers: Iterator[C]) -> Iterator[F]:
        # todo template method to obtain containers?

        for container in containers:

            # if track is observed in current iteration, update its last observed frame
            self._merged_last_track_frame.update(self._get_last_track_frames(container))
            self._unfinished_containers[container] = self._get_unfinished_tracks(
                container
            )

            # update unfinished track ids of previously tracked containers
            # if containers have no pending tracks, make ready for finishing
            newly_finished_tracks = self._get_newly_finished_tracks(container)
            newly_discarded_tracks = self._get_newly_discarded_tracks(container)
            self._discarded_tracks.update(newly_discarded_tracks)

            ready_containers: list[C] = []
            for c, track_ids in self._unfinished_containers.items():
                track_ids.difference_update(newly_finished_tracks)
                track_ids.difference_update(newly_discarded_tracks)

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
        keep = self._keep_discarded
        discarded = self._discarded_tracks
        finished_containers: list[F] = [
            self._finish(c, is_last, discarded, keep) for c in containers
        ]

        # the last frame of the observed tracks have been marked
        # track ids no longer required in _merged_last_track_frame
        cleanup_track_ids = (
            set()
            .union(*(self._get_observed_tracks(c) for c in containers))
            .union(*(self._get_newly_finished_tracks(c) for c in containers))
            .union(*(self._get_newly_discarded_tracks(c) for c in containers))
        )
        self._merged_last_track_frame = {
            track_id: frame_no
            for track_id, frame_no in self._merged_last_track_frame.items()
            if track_id not in cleanup_track_ids
        }
        self._discarded_tracks.difference_update(cleanup_track_ids)

        return finished_containers
