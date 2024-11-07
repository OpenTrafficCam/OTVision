from abc import ABC, abstractmethod
from typing import Generic, Iterator, TypeVar

from OTVision.track.data import FinishedFrame, Frame, IsLastFrame, TrackedFrame, TrackId

S = TypeVar("S")  # Source type (e.g., Path, URL, str, etc.)
# -> would look nicer in python 3.12

ID_GENERATOR = Iterator[int]


class Tracker(ABC, Generic[S]):

    @abstractmethod
    def track(
        self,
        frames: Iterator[Frame[S]],
    ) -> Iterator[TrackedFrame[S]]:
        """
        Processes a stream of frames to perform object tracking.

        S: Generic type of Frame source
        frames (Iterable[Frame[S]]):
            An iterable of frames containing detections to be tracked.
        return: An iterable of tracked frames with updated tracking information.
        """
        pass


class BufferedFinishedFramesTracker(Tracker[S]):
    # structure is very similar to BufferedFinishedChunksTracker,
    # todo maybe extract common superclass

    def __init__(self, tracker: Tracker[S]):
        super().__init__()
        self._tracker = tracker
        self._unfinished_frames: dict[TrackedFrame, set[TrackId]] = dict()
        self._merged_last_track_frame: dict[TrackId, int] = dict()

    def track(self, frames: Iterator[Frame[S]]) -> Iterator[FinishedFrame[S]]:
        for frame in self._tracker.track(frames):

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
