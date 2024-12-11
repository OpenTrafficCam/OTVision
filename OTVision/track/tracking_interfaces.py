from abc import ABC, abstractmethod
from typing import Generic, Iterator, TypeVar

from OTVision.track.data import (
    FinishedFrame,
    Frame,
    IsLastFrame,
    TrackedFrame,
    TrackId,
    UnfinishedTracksBuffer,
)

S = TypeVar("S")  # Source type (e.g., Path, URL, str, etc.)
# -> would look nicer in python 3.12

ID_GENERATOR = Iterator[TrackId]


class Tracker(ABC, Generic[S]):

    def track(
        self, frames: Iterator[Frame[S]], id_generator: ID_GENERATOR
    ) -> Iterator[TrackedFrame[S]]:
        for frame in frames:
            yield self.track_frame(frame, id_generator)

    @abstractmethod
    def track_frame(
        self,
        frame: Frame[S],
        id_generator: ID_GENERATOR,
    ) -> TrackedFrame[S]:
        pass


class UnfinishedFramesBuffer(UnfinishedTracksBuffer[TrackedFrame[S], FinishedFrame[S]]):

    def __init__(self, tracker: Tracker[S]):
        super().__init__()
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

    def _finish(
        self, container: TrackedFrame[S], is_last: IsLastFrame
    ) -> FinishedFrame[S]:
        return container.finish(is_last)


class OldBufferedFinishedFramesTracker(Tracker[S]):
    # structure is very similar to BufferedFinishedChunksTracker,
    # todo maybe extract common superclass

    def __init__(self, tracker: Tracker[S]):
        super().__init__()
        self._tracker = tracker
        self._unfinished_frames: dict[TrackedFrame, set[TrackId]] = dict()
        self._merged_last_track_frame: dict[TrackId, int] = dict()

    def track(
        self,
        frames: Iterator[Frame[S]],
        id_generator: ID_GENERATOR,
    ) -> Iterator[FinishedFrame[S]]:  # todo implement track_frame instead of track
        for frame in self._tracker.track(frames, id_generator):

            self._merged_last_track_frame.update(
                {o: frame.no for o in frame.observed_tracks}
            )
            self._unfinished_frames[frame] = frame.unfinished_tracks

            # update unfinished track ids of previously tracked frames
            # if frames have no pending tracks, make ready for finishing
            newly_finished_tracks = frame.finished_tracks
            ready_frames: list[TrackedFrame] = []
            for uframe, track_ids in self._unfinished_frames.items():
                track_ids.difference_update(newly_finished_tracks)

                if len(track_ids) == 0:
                    ready_frames.append(uframe)
                    del self._unfinished_frames[uframe]

            finished_frames = self._finish_frames(ready_frames)
            yield from finished_frames

        # finish remaining frames with pending tracks
        remaining_frames = list(self._unfinished_frames.keys())
        self._unfinished_frames = dict()

        finished_frames = self._finish_frames(remaining_frames)
        self._merged_last_track_frame = dict()
        yield from finished_frames

    def _finish_frames(self, frames: list[TrackedFrame]) -> list[FinishedFrame]:
        # TODO sort finished frames by start date?
        is_last: IsLastFrame = (
            lambda frame_no, track_id: frame_no
            == self._merged_last_track_frame[track_id]
        )
        finished_frames = [f.finish(is_last) for f in frames]

        # the last frame of the observed tracks have been marked
        # track ids no longer required in _merged_last_track_frame
        finished_tracks = set().union(*(f.observed_tracks for f in frames))
        self._merged_last_track_frame = {
            track_id: frame_no
            for track_id, frame_no in self._merged_last_track_frame.items()
            if track_id not in finished_tracks
        }

        return finished_frames
