from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Tuple

from OTVision import dataformat, version
from OTVision.dataformat import (
    CLASS,
    CONFIDENCE,
    DATA,
    DATE_FORMAT,
    DETECTIONS,
    FRAME,
    INPUT_FILE_PATH,
    INTERPOLATED_DETECTION,
    METADATA,
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
from OTVision.helpers.files import get_metadata, read_json

MISSING_START_DATE = datetime(1900, 1, 1)


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


@dataclass(frozen=True)
class FrameGroup:
    frames: list[Frame]
    order_key: str

    def start_date(self) -> datetime:
        return self.frames[0].occurrence

    def end_date(self) -> datetime:
        return self.frames[-1].occurrence

    def merge(self, other: "FrameGroup") -> "FrameGroup":
        if self.start_date() < other.start_date():
            return self._merge(self, other)
        else:
            return self._merge(other, self)

    def _merge(self, first: "FrameGroup", second: "FrameGroup") -> "FrameGroup":
        all_frames: list[Frame] = []
        all_frames.extend(first.frames)
        last_frame_number = all_frames[-1].frame
        for frame in second.frames:
            last_frame_number = last_frame_number + 1
            all_frames.append(frame.derive_frame_number(last_frame_number))
        return FrameGroup(all_frames, self.order_key)

    def get_existing_output_files(self, with_suffix: str) -> list[Path]:
        output_files = set(
            [frame.get_output_file(with_suffix=with_suffix) for frame in self.frames]
        )
        existing_files = [file for file in output_files if file.is_file()]
        return existing_files

    def update_metadata(
        self, metadata: dict[str, dict], tracker_data: dict[str, dict]
    ) -> dict[str, dict]:
        for filepath in metadata.keys():
            metadata[filepath][OTTRACK_VERSION] = version.ottrack_version()
            metadata[filepath][dataformat.TRACKING] = {
                dataformat.OTVISION_VERSION: version.otvision_version(),
                dataformat.FIRST_TRACKED_VIDEO_START: self.start_date().timestamp(),
                dataformat.LAST_TRACKED_VIDEO_END: self.end_date().timestamp(),
                dataformat.TRACKER: tracker_data,
            }
        return metadata

    def to_dict(self) -> dict:
        return {
            DATA: {frame.frame: frame.to_dict() for frame in self.frames},
        }


class Splitter:
    def split(self, tracks: dict[str, dict]) -> dict[str, list[dict]]:
        detections = self.flatten(tracks)
        detections.sort(
            key=lambda detection: (
                detection[INPUT_FILE_PATH],
                detection[FRAME],
                detection[TRACK_ID],
            )
        )
        current_group_detections: list[dict] = []
        current_input_path = ""
        groups: dict[str, list[dict]] = {}
        frame_offset = 0
        for detection in detections:
            if detection[INPUT_FILE_PATH] != current_input_path:
                if current_input_path:
                    groups[current_input_path] = current_group_detections
                current_group_detections = []
                current_input_path = detection[INPUT_FILE_PATH]
                frame_offset = detection[FRAME] - 1
            detection[FRAME] = detection[FRAME] - frame_offset
            current_group_detections.append(detection)
        groups[current_input_path] = current_group_detections
        return groups

    def flatten(self, frames: dict[str, dict]) -> list[dict]:
        detections = []
        for track in frames.values():
            for detection in track.values():
                detections.append(detection)
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


class FrameGroupParser:
    def __init__(self, input_file_path: Path, recorded_start_date: datetime) -> None:
        self.input_file_path = input_file_path
        self.recorded_start_date = recorded_start_date

    def convert(self, input: dict[int, dict[str, Any]]) -> FrameGroup:
        detection_parser = DetectionParser()
        frames = []
        for key, value in input.items():
            occurrence: datetime = parse_datetime(value[OCCURRENCE])
            data_detections = value[DETECTIONS]
            detections = detection_parser.convert(data_detections)
            parsed_frame = Frame(
                int(key),
                occurrence=occurrence,
                input_file_path=self.input_file_path,
                detections=detections,
            )
            frames.append(parsed_frame)

        frames.sort(key=lambda frame: (frame.occurrence, frame.frame))
        return FrameGroup(frames, order_key=self.order_key())

    def order_key(self) -> str:
        return self.input_file_path.parent.as_posix()


@dataclass(frozen=True)
class PreprocessResult:
    frame_groups: list[FrameGroup]
    metadata: dict[str, dict]


class Preprocess:
    """Preprocess otdet files before running track. Input files belonging to the same
    recording will be merged together. The time gap to separate two recordings from
    each other is defined by `self.time_without_frames`.

    Returns:
        Preprocess: preprocessor for tracking
    """

    def __init__(self, time_without_frames: timedelta = timedelta(minutes=1)) -> None:
        self.time_without_frames = time_without_frames

    def run(self, files: list[Path]) -> PreprocessResult:
        """Read all input files, parse the content and merge the frame groups belonging
        together.

        Args:
            files (list[Path]): list of input files

        Returns:
            PreprocessResult: merged frame groups and metadata per file
        """
        input_data = {}
        for file in files:
            input = read_json(file)
            input_data[file] = input
        groups, metadata = self.process(input_data)
        return PreprocessResult(frame_groups=groups, metadata=metadata)

    def process(
        self, input: dict[Path, dict]
    ) -> Tuple[list[FrameGroup], dict[str, dict]]:
        """Parse input and merge frame groups belonging together.

        Args:
            input (dict[Path, dict]): input by input file path

        Returns:
            Tuple[list[FrameGroup], dict[str, dict]]: parsed and merged frame groups
        """
        all_groups, metadata = self._parse_frame_groups(input)
        if len(all_groups) == 0:
            return [], metadata
        return self._merge_groups(all_groups), metadata

    def _parse_frame_groups(
        self, input: dict[Path, dict]
    ) -> Tuple[list[FrameGroup], dict[str, dict]]:
        """Parse input to frame groups. Every input file belongs to one frame group.

        Args:
            input (dict[Path, dict]): read in input (otdet)

        Returns:
            Tuple[list[FrameGroup], dict[str, dict]]: parsed input and metadata per file
        """
        all_groups: list[FrameGroup] = []
        metadata: dict[str, dict] = {}
        for file_path, recording in input.items():
            file_metadata = get_metadata(otdict=recording)
            metadata[file_path.as_posix()] = file_metadata
            start_date: datetime = self.extract_start_date_from(recording)

            data: dict[int, dict[str, Any]] = recording[DATA]
            frame_group = FrameGroupParser(
                file_path, recorded_start_date=start_date
            ).convert(data)
            all_groups.append(frame_group)
        return all_groups, metadata

    def _merge_groups(self, all_groups: list[FrameGroup]) -> list[FrameGroup]:
        """Merge frame groups whose start and end times are close to each other. Close
        is defined by `self.time_without_frames`.

        Args:
            all_groups (list[FrameGroup]): list of frame groups to merge

        Returns:
            list[FrameGroup]: list of merged frame groups
        """
        merged_groups = []
        last_group = all_groups[0]
        for current_group in all_groups[1:]:
            if (
                current_group.start_date() - last_group.end_date()
            ) <= self.time_without_frames:
                last_group = last_group.merge(current_group)
            else:
                merged_groups.append(last_group)
                last_group = current_group
        merged_groups.append(last_group)
        return merged_groups

    def extract_start_date_from(self, recording: dict) -> datetime:
        if RECORDED_START_DATE in recording[METADATA][VIDEO].keys():
            recorded_start_date = recording[METADATA][VIDEO][RECORDED_START_DATE]
            return parse_datetime(recorded_start_date)
        return MISSING_START_DATE
