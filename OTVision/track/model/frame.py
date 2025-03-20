from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Generic, Optional, Sequence, TypeVar

from PIL.Image import Image

from OTVision.dataformat import FRAME, OCCURRENCE, TRACK_ID
from OTVision.domain.detection import (
    Detection,
    FinishedDetection,
    TrackedDetection,
    TrackId,
)

FrameNo = int
S = TypeVar("S")


@dataclass(frozen=True)
class Frame(Generic[S]):
    """Frame metadata, optional image and respective detections.

    Attributes:
        no (FrameNo): Frame number.
        occurrence (datetime): Time stamp, at which frame was recorded.
        source (S): Generic source from where frame was obtained, e.g. video file path.
        detections (Sequence[Detection]): A sequence of Detections occurring in frame.
        image (Optional[Image]): Optional image data of frame.
    """

    no: FrameNo
    occurrence: datetime
    source: S
    detections: Sequence[Detection]
    image: Optional[Image]


IsLastFrame = Callable[[FrameNo, TrackId], bool]


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
        """
        Derive observed and unfinished tracks from tracked detections and finished
        track information.
        """
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

    def to_detection_dicts(self) -> list[dict]:
        frame_metadata = {FRAME: self.no, OCCURRENCE: self.occurrence.timestamp()}

        # add frame metadata to each detection dict
        detection_dict_list = [
            {**detection.to_dict(), **frame_metadata} for detection in self.detections
        ]

        detection_dict_list.sort(key=lambda det: det[TRACK_ID])
        return detection_dict_list
