from abc import ABC, abstractmethod
from typing import Generic, Iterator, TypeVar

from OTVision.track.model.frame import (
    FinishedFrame,
    Frame,
    FrameNo,
    IsLastFrame,
    TrackedFrame,
    TrackId,
)

S = TypeVar("S")  # Source type (e.g., Path, URL, str, etc.)
# -> would look nicer in python 3.12

ID_GENERATOR = Iterator[TrackId]


class Tracker(ABC, Generic[S]):
    """Tracker interface for processing a stream of Frames
    to add tracking information, creating a lazy stream (generator)
    of TrackedFrames.

    Implementing class can specify template method:
    track_frame for processing a single frame.

    Args:
        Generic (S): generic type of Frame source (e.g. file path, or stream url)
    """

    def track(
        self, frames: Iterator[Frame[S]], id_generator: ID_GENERATOR
    ) -> Iterator[TrackedFrame[S]]:
        """Process the given stream of Frames,
        yielding TrackedFrames one by one as a lazy stream of TrackedFrames.

        Args:
            frames (Iterator[Frame[S]]): (lazy) stream of Frames
                with untracked Detections.
            id_generator (ID_GENERATOR): provider of new (unique) track ids.

        Yields:
            Iterator[TrackedFrame[S]]: (lazy) stream of TrackedFrames with
                TrackedDetections
        """
        for frame in frames:
            yield self.track_frame(frame, id_generator)

    @abstractmethod
    def track_frame(
        self,
        frame: Frame[S],
        id_generator: ID_GENERATOR,
    ) -> TrackedFrame[S]:
        """Process single Frame with untracked Detections,
        by adding tracking information,
        creating a TrackedFrame with TrackedDetections.

        Args:
            frame (Frame[S]): the Frame (with source S) to be tracked.
            id_generator (ID_GENERATOR): provider of new (unique) track ids.

        Returns:
            TrackedFrame[S]: TrackedFrame with TrackedDetections
        """
        pass


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
        self._merged_last_track_frame: dict[TrackId, FrameNo] = dict()
        self._discarded_tracks: set[TrackId] = set()

    @abstractmethod
    def _get_last_track_frames(self, container: C) -> dict[TrackId, FrameNo]:
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
        # TODO template method to obtain containers?

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


class UnfinishedFramesBuffer(UnfinishedTracksBuffer[TrackedFrame[S], FinishedFrame[S]]):
    """UnfinishedTracksBuffer implementation for Frames as Detection container."""

    def __init__(self, tracker: Tracker[S], keep_discarded: bool = False):
        super().__init__(keep_discarded)
        self._tracker = tracker

    def track(
        self, frames: Iterator[Frame[S]], id_generator: ID_GENERATOR
    ) -> Iterator[FinishedFrame[S]]:
        tracked_frame_stream = self._tracker.track(frames, id_generator)
        return self.track_and_finish(tracked_frame_stream)

    def _get_last_track_frames(self, container: TrackedFrame[S]) -> dict[TrackId, int]:
        return {o: container.no for o in container.observed_tracks}

    def _get_unfinished_tracks(self, container: TrackedFrame[S]) -> set[TrackId]:
        return container.unfinished_tracks

    def _get_observed_tracks(self, container: TrackedFrame[S]) -> set[TrackId]:
        return container.observed_tracks

    def _get_newly_finished_tracks(self, container: TrackedFrame[S]) -> set[TrackId]:
        return container.finished_tracks

    def _get_newly_discarded_tracks(self, container: TrackedFrame[S]) -> set[TrackId]:
        return container.discarded_tracks

    def _finish(
        self,
        container: TrackedFrame[S],
        is_last: IsLastFrame,
        discarded_tracks: set[TrackId],
        keep_discarded: bool,
    ) -> FinishedFrame[S]:
        return container.finish(is_last, discarded_tracks, keep_discarded)
