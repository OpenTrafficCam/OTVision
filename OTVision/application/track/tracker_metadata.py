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

"""Tracker metadata written to the ``tracking.tracker`` section of ``.ottrk``."""

from dataclasses import dataclass, field

from OTVision import dataformat
from OTVision.application.config import BotSortTrackerParam, TrackConfig
from OTVision.domain.tracker import TrackerType


@dataclass(frozen=True)
class TrackerMetadata:
    """Identity, lifecycle, and tuning of the tracker that produced a file.

    ``params`` holds whatever the tracker reports about its own tuning: the
    sigma thresholds for IOU, the effective Ultralytics args for BoT-SORT.

    Attributes:
        name: Tracker name as written to ``.ottrk``.
        t_min: Minimum track span in frames.
        t_miss_max: Maximum missing frames before finish/discard.
        params: Tracker-specific parameters, already nested under their key.
    """

    name: str
    t_min: int
    t_miss_max: int
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to the ``tracking.tracker`` mapping.

        Returns:
            dict: Tracker metadata mapping.
        """
        return {
            dataformat.NAME: self.name,
            dataformat.T_MIN: self.t_min,
            dataformat.T_MISS_MAX: self.t_miss_max,
            **self.params,
        }


def tracker_metadata_of(
    track_config: TrackConfig,
    resolved_botsort_params: dict[str, BotSortTrackerParam] | None = None,
) -> TrackerMetadata:
    """Build tracker metadata for the tracker selected in ``track_config``.

    This is the only place that maps a tracker type onto its metadata shape.

    Args:
        track_config (TrackConfig): Track configuration.
        resolved_botsort_params (dict[str, BotSortTrackerParam] | None): Effective
            BoT-SORT args (defaults + overrides + FPS-derived ``track_buffer``).
            Required in BoT-SORT mode, ignored otherwise.

    Returns:
        TrackerMetadata: Metadata of the selected tracker.
    """
    lifecycle = track_config.lifecycle
    if track_config.tracker_type is TrackerType.BOTSORT:
        # IOU thresholds are not used by BoT-SORT; omit them from metadata.
        params = (
            {dataformat.TRACKER_PARAMS: resolved_botsort_params}
            if resolved_botsort_params
            else {}
        )
        return TrackerMetadata(
            name="BoTSORT",
            t_min=lifecycle.t_min,
            t_miss_max=lifecycle.t_miss_max,
            params=params,
        )
    iou = track_config.iou
    return TrackerMetadata(
        name="IOU",
        t_min=lifecycle.t_min,
        t_miss_max=lifecycle.t_miss_max,
        params={
            dataformat.SIGMA_L: iou.sigma_l,
            dataformat.SIGMA_H: iou.sigma_h,
            dataformat.SIGMA_IOU: iou.sigma_iou,
        },
    )
