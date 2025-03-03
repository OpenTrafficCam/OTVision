from dataclasses import dataclass

from OTVision.dataformat import (
    CLASS,
    CONFIDENCE,
    FINISHED,
    FIRST,
    INTERPOLATED_DETECTION,
    TRACK_ID,
    H,
    W,
    X,
    Y,
)

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

    def to_otdet(self) -> dict:
        return {
            CLASS: self.label,
            CONFIDENCE: self.conf,
            X: self.x,
            Y: self.y,
            W: self.w,
            H: self.h,
        }


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

    def as_last_detection(self, is_discarded: bool) -> "FinishedDetection":
        return FinishedDetection.from_tracked_detection(
            self, is_last=True, is_discarded=is_discarded
        )

    def as_intermediate_detection(self, is_discarded: bool) -> "FinishedDetection":
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

    def to_dict(self) -> dict:
        return {
            CLASS: self.label,
            CONFIDENCE: self.conf,
            X: self.x,
            Y: self.y,
            W: self.w,
            H: self.h,
            INTERPOLATED_DETECTION: False,
            FIRST: self.is_first,
            FINISHED: self.is_last,
            TRACK_ID: self.track_id,
        }
