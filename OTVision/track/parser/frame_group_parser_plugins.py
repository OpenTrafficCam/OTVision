import re
from datetime import datetime, timedelta
from pathlib import Path

from OTVision import dataformat, version
from OTVision.application.config import TrackConfig
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.dataformat import (
    EXPECTED_DURATION,
    FILENAME,
    FIRST_TRACKED_VIDEO_START,
    LAST_TRACKED_VIDEO_END,
    LENGTH,
    OTTRACK_VERSION,
    OTVISION_VERSION,
    RECORDED_START_DATE,
    TRACKER,
    TRACKING,
    VIDEO,
)
from OTVision.detect.otdet import parse_video_length
from OTVision.helpers.files import (
    FULL_FILE_NAME_PATTERN,
    HOSTNAME,
    InproperFormattedFilename,
    read_json_bz2_metadata,
)
from OTVision.track.model.filebased.frame_group import FrameGroup, FrameGroupParser
from OTVision.track.parser.chunk_parser_plugins import parse_datetime

MISSING_START_DATE = datetime(1900, 1, 1)
MISSING_EXPECTED_DURATION = timedelta(minutes=15)


class TimeThresholdFrameGroupParser(FrameGroupParser):
    @property
    def config(self) -> TrackConfig:
        return self._get_current_config.get().track

    @property
    def _tracker_data(self) -> dict:
        return tracker_metadata(
            sigma_l=self.config.sigma_l,
            sigma_h=self.config.sigma_h,
            sigma_iou=self.config.sigma_iou,
            t_min=self.config.t_min,
            t_miss_max=self.config.t_miss_max,
        )

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
        video_name = Path(file_metadata[VIDEO][FILENAME]).name
        match = re.search(
            FULL_FILE_NAME_PATTERN,
            video_name,
        )
        if match:
            return match.group(HOSTNAME)

        raise InproperFormattedFilename(f"Could not parse {video_name}.")

    def extract_start_date_from(self, metadata: dict) -> datetime:
        if RECORDED_START_DATE in metadata[VIDEO].keys():
            recorded_start_date = metadata[VIDEO][RECORDED_START_DATE]
            return parse_datetime(recorded_start_date)
        return MISSING_START_DATE

    def extract_expected_duration_from(self, metadata: dict) -> timedelta:
        if EXPECTED_DURATION in metadata[VIDEO].keys():
            if expected_duration := metadata[VIDEO][EXPECTED_DURATION]:
                return timedelta(seconds=int(expected_duration))
        return self.parse_video_length(metadata)

    def parse_video_length(self, metadata: dict) -> timedelta:
        video_length = metadata[VIDEO][LENGTH]
        return parse_video_length(video_length)

    def update_metadata(self, frame_group: FrameGroup) -> dict[Path, dict]:
        metadata_by_file = dict(frame_group.metadata_by_file)
        for filepath in frame_group.files:
            metadata = metadata_by_file[filepath]
            metadata[OTTRACK_VERSION] = version.ottrack_version()
            metadata[TRACKING] = {
                OTVISION_VERSION: version.otvision_version(),
                FIRST_TRACKED_VIDEO_START: frame_group.start_date.timestamp(),
                LAST_TRACKED_VIDEO_END: frame_group.end_date.timestamp(),
                TRACKER: self._tracker_data,
            }

        return metadata_by_file

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


def tracker_metadata(
    sigma_l: float, sigma_h: float, sigma_iou: float, t_min: float, t_miss_max: float
) -> dict:
    return {
        dataformat.NAME: "IOU",
        dataformat.SIGMA_L: sigma_l,
        dataformat.SIGMA_H: sigma_h,
        dataformat.SIGMA_IOU: sigma_iou,
        dataformat.T_MIN: t_min,
        dataformat.T_MISS_MAX: t_miss_max,
    }
