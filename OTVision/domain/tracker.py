# Copyright (C) 2022 OpenTrafficCam Contributors
# <https://github.com/OpenTrafficCam
# <team@opentrafficcam.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
