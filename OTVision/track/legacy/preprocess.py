import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from OTVision import dataformat, version
from OTVision.dataformat import (
    CLASS,
    CONFIDENCE,
    DATA,
    DATE_FORMAT,
    DETECTIONS,
    EXPECTED_DURATION,
    FILENAME,
    FRAME,
    INPUT_FILE_PATH,
    INTERPOLATED_DETECTION,
    OCCURRENCE,
    OTTRACK_VERSION,
    RECORDED_START_DATE,
    TRACK_ID,
    VIDEO,
    H,
    W,
    X,
    Y,
)
from OTVision.helpers.date import (
    parse_date_string_to_utc_datime,
    parse_timestamp_string_to_utc_datetime,
)
from OTVision.helpers.files import (
    FULL_FILE_NAME_PATTERN,
    HOSTNAME,
    InproperFormattedFilename,
    read_json,
    read_json_bz2_metadata,
)

MISSING_START_DATE = datetime(1900, 1, 1)
MISSING_EXPECTED_DURATION = timedelta(minutes=15)


def parse_datetime(date: str | float) -> datetime:
    """Parse a date string or timestamp to a datetime with UTC as timezone.

    Args:
        date (str | float): the date to parse

    Returns:
        datetime: the parsed datetime object with UTC set as timezone
    """
    if isinstance(date, str) and ("-" in date):
        return parse_date_string_to_utc_datime(date, DATE_FORMAT)
    return parse_timestamp_string_to_utc_datetime(date)


@dataclass(frozen=True, repr=True)
class Detection:
    """
    Data class which contains information for a single detection.
    """

    label: str
    conf: float
    x: float
    y: float
    w: float
    h: float

    def to_dict(self, frame: int, occurrence: datetime, input_file_path: str) -> dict:
        return {
            CLASS: self.label,
            CONFIDENCE: self.conf,
            X: self.x,
            Y: self.y,
            W: self.w,
            H: self.h,
            FRAME: frame,
            OCCURRENCE: occurrence.timestamp(),
            INPUT_FILE_PATH: input_file_path,
            INTERPOLATED_DETECTION: False,
        }

    def to_otdet(self) -> dict:
        return {
            CLASS: self.label,
            CONFIDENCE: self.conf,
            X: self.x,
            Y: self.y,
            W: self.w,
            H: self.h,
        }


@dataclass(frozen=True)
class Frame:
    frame: int
    occurrence: datetime
    input_file_path: Path
    detections: list[Detection]

    def to_dict(self) -> dict:
        return {
            FRAME: self.frame,
            OCCURRENCE: self.occurrence.timestamp(),
            INPUT_FILE_PATH: self.input_file_path.as_posix(),
            DETECTIONS: [
                detection.to_dict(
                    self.frame, self.occurrence, self.input_file_path.as_posix()
                )
                for detection in self.detections
            ],
        }

    def derive_frame_number(self, new_frame_number: int) -> "Frame":
        return Frame(
            new_frame_number, self.occurrence, self.input_file_path, self.detections
        )

    def get_output_file(self, with_suffix: str) -> Path:
        return self.input_file_path.with_suffix(with_suffix)


class FrameGroup:
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        hostname: str,
        file: Path | None,
        metadata: dict | None,
    ) -> None:
        self._files_metadata: dict[str, dict] = dict()
        self._files: list[Path] = []
        if (file is not None) and (metadata is not None):
            self._files_metadata[file.as_posix()] = metadata
            self._files.append(file)

        self.hostname = hostname
        self._start_date = start_date
        self._end_date = end_date

    @property
    def files(self) -> list[Path]:
        return self._files

    def metadata_for(self, file: Path | str) -> dict:
        if isinstance(file, str):
            return self._files_metadata[file]
        else:
            return self._files_metadata[file.as_posix()]

    def start_date(self) -> datetime:
        return self._start_date

    def end_date(self) -> datetime:
        return self._end_date

    def merge(self, other: "FrameGroup") -> "FrameGroup":
        if self.start_date() < other.start_date():
            return self._merge(self, other)
        else:
            return self._merge(other, self)

    def _merge(self, first: "FrameGroup", second: "FrameGroup") -> "FrameGroup":
        if first.hostname != second.hostname:
            raise ValueError("Hostname of FrameGroups does not match")
        merged = FrameGroup(
            start_date=first._start_date,
            end_date=second._end_date,
            hostname=self.hostname,
            file=None,
            metadata=None,
        )

        merged._files_metadata.update(first._files_metadata)
        merged._files_metadata.update(second._files_metadata)
        merged._files += first.files
        merged._files += second.files

        return merged

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"{self._start_date} - {self._end_date}"

    def update_metadata(self, tracker_data: dict[str, dict]) -> None:
        for filepath in self.files:
            metadata = self.metadata_for(filepath)
            metadata[OTTRACK_VERSION] = version.ottrack_version()
            metadata[dataformat.TRACKING] = {
                dataformat.OTVISION_VERSION: version.otvision_version(),
                dataformat.FIRST_TRACKED_VIDEO_START: self.start_date().timestamp(),
                dataformat.LAST_TRACKED_VIDEO_END: self.end_date().timestamp(),
                dataformat.TRACKER: tracker_data,
            }


@dataclass(frozen=True)
class FrameChunk:
    file: Path
    frames: list[Frame]

    def start_date(self) -> datetime:
        return self.frames[0].occurrence

    def end_date(self) -> datetime:
        return self.frames[-1].occurrence

    def last_frame_id(self) -> int:
        return self.frames[-1].frame

    def get_existing_output_files(self, with_suffix: str) -> list[Path]:
        output_files = set(
            [frame.get_output_file(with_suffix=with_suffix) for frame in self.frames]
        )
        existing_files = [file for file in output_files if file.is_file()]
        return existing_files

    def to_dict(self) -> dict:
        return {
            DATA: {frame.frame: frame.to_dict() for frame in self.frames},
        }

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return str(self.file)


class FrameIndexer:
    def reindex(self, frames: dict[str, dict], frame_offset: int) -> list[dict]:
        detections = []
        for track in frames.values():
            for detection in track.values():
                # Take into account that consecutive tracks over more than one
                # video must have their frame reset to one when splitting.
                # This is done by taking the frame_offset into account.
                detection[FRAME] = detection[FRAME] - frame_offset
                detections.append(detection)

        if len(detections) == 0:
            return []

        assert len({detection[INPUT_FILE_PATH] for detection in detections}) == 1

        detections.sort(
            key=lambda detection: (
                detection[INPUT_FILE_PATH],
                detection[FRAME],
                detection[TRACK_ID],
            )
        )

        return detections


class DetectionParser:
    def convert(self, data_detections: list[dict[str, str]]) -> list[Detection]:
        detections: list[Detection] = []
        for detection in data_detections:
            detected_item = Detection(
                detection[CLASS],
                float(detection[CONFIDENCE]),
                float(detection[X]),
                float(detection[Y]),
                float(detection[W]),
                float(detection[H]),
            )
            detections.append(detected_item)
        return detections


class FrameChunkParser:
    @staticmethod
    def parse(
        file_path: Path,
        frame_offset: int = 0,
    ) -> FrameChunk:
        input: dict[int, dict[str, Any]] = read_json(file_path)[DATA]
        return FrameChunkParser.convert(input, file_path, frame_offset)

    @staticmethod
    def convert(
        input: dict[int, dict[str, Any]],
        file_path: Path,
        frame_offset: int = 0,
    ) -> FrameChunk:
        detection_parser = DetectionParser()
        frames: list[Frame] = []
        for key, value in input.items():
            occurrence: datetime = parse_datetime(value[OCCURRENCE])
            data_detections = value[DETECTIONS]
            detections = detection_parser.convert(data_detections)
            parsed_frame = Frame(
                int(key) + frame_offset,
                occurrence=occurrence,
                input_file_path=file_path,
                detections=detections,
            )
            frames.append(parsed_frame)

        frames.sort(key=lambda frame: (frame.occurrence, frame.frame))
        return FrameChunk(file_path, frames)


class Preprocess:
    """Preprocess otdet file metadata (recording time interval) before running track.
    Input files belonging to the same recording will be merged together.
    The time gap to separate two recordings from each other is defined by
    `self.time_without_frames`.

    Returns:
        Preprocess: preprocessor for tracking
    """

    def __init__(self, time_without_frames: timedelta = timedelta(minutes=1)) -> None:
        self.time_without_frames = time_without_frames

    def run(self, files: list[Path]) -> list[FrameGroup]:
        """Read metadata of all input files,
        parse the content and merge the frame groups belonging together.

        Args:
            files (list[Path]): list of input files

        Returns:
            list[FrameGroup]: merged frame groups sorted by start date
        """

        groups = self.process(self._read_input(files))
        return sorted(groups, key=lambda r: r.start_date())

    def process(self, input: dict[Path, dict]) -> list[FrameGroup]:
        """Process given otdet files:
        Create FrameGroup for each file then merge frame groups belonging together.

        Args:
            files (list[Path]): list of file paths

        Returns:
            list[FrameGroup]: parsed and merged frame groups
        """
        all_groups = [
            self._parse_frame_group(path, metadata) for path, metadata in input.items()
        ]
        if len(all_groups) == 0:
            return []
        return self._merge_groups(all_groups)

    def _read_input(self, files: list[Path]) -> dict[Path, dict]:
        return {path: read_json_bz2_metadata(path) for path in files}

    def _parse_frame_group(self, file_path: Path, metadata: dict) -> FrameGroup:
        """Read and parse metadata of the given file to a FrameGroup
        covering the recording time interval defined by:
        - the recorded start date and
        - the expected duration given in the metadata

        Args:
            file_path (Path): path of otdet file
            metadata (dict): metadata of otdet file

        Returns:
            list[FrameGroup]: parsed input and metadata per file
        """

        start_date: datetime = self.extract_start_date_from(metadata)
        duration: timedelta = self.extract_expected_duration_from(metadata)
        end_date: datetime = start_date + duration
        hostname = self.get_hostname(metadata)

        return FrameGroup(
            start_date=start_date,
            end_date=end_date,
            file=file_path,
            metadata=metadata,
            hostname=hostname,
        )

    @staticmethod
    def get_hostname(file_metadata: dict) -> str:
        """Retrieve hostname from the given file metadata.

        Args:
            file_metadata (dict): metadata content.

        Raises:
            InproperFormattedFilename: if the filename is not formatted as expected, an
                exception will be raised.

        Returns:
            str: the hostname
        """
        video_name = Path(file_metadata[VIDEO][FILENAME]).name
        match = re.search(
            FULL_FILE_NAME_PATTERN,
            video_name,
        )
        if match:
            return match.group(HOSTNAME)

        raise InproperFormattedFilename(
            f"Could not parse {video_name} with pattern: {FULL_FILE_NAME_PATTERN}."
        )

    def _merge_groups(self, all_groups: list[FrameGroup]) -> list[FrameGroup]:
        """Merge frame groups whose start and end times are close to each other. Close
        is defined by `self.time_without_frames`.

        Args:
            all_groups (list[FrameGroup]): list of frame groups to merge

        Returns:
            list[FrameGroup]: list of merged frame groups
        """
        assert len(all_groups) >= 1

        merged_groups = []
        sorted_groups = sorted(all_groups, key=lambda group: group.start_date())
        last_group = sorted_groups[0]
        for current_group in sorted_groups[1:]:
            if last_group.hostname != current_group.hostname:
                merged_groups.append(last_group)
                last_group = current_group
            elif (
                timedelta(seconds=0)
                <= (current_group.start_date() - last_group.end_date())
                <= self.time_without_frames
            ):
                last_group = last_group.merge(current_group)
            else:
                merged_groups.append(last_group)
                last_group = current_group
        merged_groups.append(last_group)
        return merged_groups

    def extract_start_date_from(self, metadata: dict) -> datetime:
        if RECORDED_START_DATE in metadata[VIDEO].keys():
            recorded_start_date = metadata[VIDEO][RECORDED_START_DATE]
            return parse_datetime(recorded_start_date)
        return MISSING_START_DATE

    def extract_expected_duration_from(self, metadata: dict) -> timedelta:
        if EXPECTED_DURATION in metadata[VIDEO].keys():
            expected_duration = metadata[VIDEO][EXPECTED_DURATION]
            return timedelta(seconds=int(expected_duration))
        return MISSING_EXPECTED_DURATION
