from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Literal, Optional, Sequence, TypedDict

from numpy import ndarray

from OTVision.dataformat import FRAME, OCCURRENCE, TRACK_ID
from OTVision.domain.detection import (
    Detection,
    FinishedDetection,
    TrackedDetection,
    TrackId,
)


class FrameKeys:
    """Keys to access Frame dictionary."""

    data: Literal["data"] = "data"
    frame: Literal["frame"] = "frame"
    source: Literal["source"] = "source"
    occurrence: Literal["occurrence"] = "occurrence"
    output: Literal["output"] = "output"


class Frame(TypedDict):
    """Frame definition.

    Attributes:
        data (Optional[ndarray]): The frame data as a numpy array, can be None.
        frame (int): The frame number.
        source (str): The source identifier of the frame.
        occurrence (datetime): Timestamp when the frame was captured/created.
    """

    data: Optional[ndarray]
    frame: int
    source: str
    output: str
    occurrence: datetime


FrameNo = int


@dataclass(frozen=True, kw_only=True)
class DetectedFrame:
    """Frame metadata, optional image and respective detections.

    Attributes:
        no (FrameNo): Frame number.
        occurrence (datetime): Time stamp, at which frame was recorded.
        source (str): Source from where frame was obtained, e.g. video file path.
        output (str): Output file name, e.g. video file name.
        detections (Sequence[Detection]): A sequence of Detections occurring in frame.
        image (Optional[ndarray]): Optional image data of frame.
    """

    no: FrameNo
    occurrence: datetime
    source: str
    output: str
    detections: Sequence[Detection]
    image: Optional[ndarray] = None

    def without_image(self) -> "DetectedFrame":
        return DetectedFrame(
            no=self.no,
            occurrence=self.occurrence,
            source=self.source,
            output=self.output,
            detections=self.detections,
            image=None,
        )


IsLastFrame = Callable[[FrameNo, TrackId], bool]


@dataclass(frozen=True, kw_only=True)
class TrackedFrame(DetectedFrame):
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
            output=self.output,
            finished_tracks=self.finished_tracks,
            detections=detections,
            image=self.image,
            discarded_tracks=discarded_tracks,
        )


@dataclass(frozen=True, kw_only=True)
class FinishedFrame(TrackedFrame):
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
