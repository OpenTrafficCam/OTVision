"""Tracker selection and lifecycle values shared across OTVision layers."""

from dataclasses import dataclass
from enum import StrEnum


class TrackerType(StrEnum):
    """Tracker implementation selected via ``TRACK.TRACKER_TYPE``."""

    IOU = "iou"
    BOTSORT = "botsort"

    @classmethod
    def values(cls) -> list[str]:
        """Return all valid tracker type values, for CLI choices and validation.

        Returns:
            list[str]: Sorted tracker type values.
        """
        return sorted(member.value for member in cls)


@dataclass(frozen=True)
class TrackerLifecycle:
    """Frame-count thresholds governing when a track is finished or discarded.

    Attributes:
        t_min: Minimum frame span a track must cover to be kept.
        t_miss_max: Consecutive missing frames tolerated before finish/discard.
    """

    t_min: int
    t_miss_max: int
