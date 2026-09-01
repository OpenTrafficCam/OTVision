from datetime import datetime, timedelta
from pathlib import Path

from OTVision.application.config import TrackConfig
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.application.track.ottrk import create_ottrk_metadata_entry
from OTVision.detect.otdet import (
    extract_expected_duration_from_otdet,
    extract_hostname_from_otdet,
    extract_start_date_from_otdet,
)
from OTVision.helpers.files import read_json_bz2_metadata
from OTVision.track.model.filebased.frame_group import FrameGroup, FrameGroupParser
from OTVision.track.tracker.tracker_plugin_botsort import (
    extract_frame_rate_from_metadata,
    resolve_botsort_tracker_params,
)

MISSING_EXPECTED_DURATION = timedelta(minutes=15)


class TimeThresholdFrameGroupParser(FrameGroupParser):
    @property
    def config(self) -> TrackConfig:
        return self._get_current_config.get().track

    def __init__(
        self,
        get_current_config: GetCurrentConfig,
        time_without_frames: timedelta = timedelta(minutes=1),
    ):
        self._get_current_config = get_current_config
        self._time_without_frames = time_without_frames
        self._id_count = 0

    def new_id(self) -> int:
        self._id_count += 1
        return self._id_count

    def parse(self, file: Path) -> FrameGroup:
        metadata = read_json_bz2_metadata(file)
        return self.convert(file, metadata)

    def convert(self, file: Path, metadata: dict) -> FrameGroup:
        start_date: datetime = self.extract_start_date_from(metadata)
        duration: timedelta = self.extract_expected_duration_from(metadata)
        end_date: datetime = start_date + duration
        hostname = self.get_hostname(metadata)

        return FrameGroup(
            id=self.new_id(),
            start_date=start_date,
            end_date=end_date,
            files=[file],
            metadata_by_file={file: metadata},
            hostname=hostname,
        )

    def get_hostname(self, file_metadata: dict) -> str:
        return extract_hostname_from_otdet(file_metadata)

    def extract_start_date_from(self, metadata: dict) -> datetime:
        return extract_start_date_from_otdet(metadata)

    def extract_expected_duration_from(self, metadata: dict) -> timedelta:
        return extract_expected_duration_from_otdet(metadata)

    def update_metadata(self, frame_group: FrameGroup) -> dict[Path, dict]:
        metadata_by_file = dict(frame_group.metadata_by_file)
        # Resolve once from the first file: GroupedFilesTracker builds one BoT-SORT
        # instance for the whole group using that first source's FPS.
        first_metadata = metadata_by_file[frame_group.files[0]]
        tracker_params = self._resolve_tracker_params_for_metadata(first_metadata)
        for filepath in frame_group.files:
            metadata = metadata_by_file[filepath]
            ottrk_metadata = create_ottrk_metadata_entry(
                start_date=frame_group.start_date,
                end_date=frame_group.end_date,
                sigma_l=self.config.sigma_l,
                sigma_h=self.config.sigma_h,
                sigma_iou=self.config.sigma_iou,
                t_min=self.config.t_min,
                t_miss_max=self.config.t_miss_max,
                tracker_type=self.config.tracker_type,
                tracker_params=tracker_params,
            )
            metadata.update(ottrk_metadata)

        return metadata_by_file

    def _resolve_tracker_params_for_metadata(
        self, otdet_metadata: dict
    ) -> dict[str, bool | int | float | str]:
        """Resolve effective BoT-SORT params for ``.ottrk`` metadata.

        Args:
            otdet_metadata (dict): Source OTDET metadata used for FPS (group-first).

        Returns:
            dict[str, bool | int | float | str]: Effective BoT-SORT params, or
            empty for IOU.
        """
        if self.config.tracker_type != "botsort":
            return {}
        fps = extract_frame_rate_from_metadata(otdet_metadata)
        if fps is None:
            raise ValueError(
                "BoT-SORT metadata requires FPS in .otdet video section "
                "to resolve effective tracker parameters."
            )
        frame_rate = max(1, int(round(fps)))
        return resolve_botsort_tracker_params(self.config.botsort, frame_rate)

    def merge(self, frame_groups: list[FrameGroup]) -> list[FrameGroup]:
        if len(frame_groups) == 0:
            return []

        merged_groups = []
        sorted_groups = sorted(frame_groups, key=lambda group: group.start_date)
        last_group = sorted_groups[0]
        for current_group in sorted_groups[1:]:
            if last_group.hostname != current_group.hostname:
                merged_groups.append(last_group)
                last_group = current_group
            elif (
                timedelta(seconds=0)
                <= (current_group.start_date - last_group.end_date)
                <= self._time_without_frames
            ):
                last_group = last_group.merge(current_group)
            else:
                merged_groups.append(last_group)
                last_group = current_group
        merged_groups.append(last_group)
        return merged_groups
