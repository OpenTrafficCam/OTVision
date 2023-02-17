from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Tuple

from OTVision.dataformat import (
    CLASS,
    CLASSIFIED,
    CONFIDENCE,
    DATA,
    DATE_FORMAT,
    FRAME,
    INPUT_FILE_PATH,
    METADATA,
    OCCURRENCE,
    RECORDED_START_DATE,
    TRACK_ID,
    VIDEO,
    H,
    W,
    X,
    Y,
)
from OTVision.helpers.files import read_json

MISSING_START_DATE = datetime(1900, 1, 1)


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
            OCCURRENCE: occurrence.strftime(DATE_FORMAT),
            INPUT_FILE_PATH: input_file_path,
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
            OCCURRENCE: self.occurrence.strftime(DATE_FORMAT),
            INPUT_FILE_PATH: self.input_file_path.as_posix(),
            CLASSIFIED: [
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
        current_video_path = ""
        groups: dict[str, list[dict]] = {}
        frame_offset = 0
        for detection in detections:
            if detection[INPUT_FILE_PATH] != current_video_path:
                if len(current_group_detections) > 0:
                    groups[current_video_path] = current_group_detections
                current_group_detections = []
                current_video_path = detection[INPUT_FILE_PATH]
                frame_offset = detection[FRAME] - 1
            detection[FRAME] = detection[FRAME] - frame_offset
            current_group_detections.append(detection)
        groups[current_video_path] = current_group_detections
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
    input_file_path: Path
    recorded_start_date: datetime

    def __init__(self, input_file_path: Path, recorded_start_date: datetime) -> None:
        self.input_file_path = input_file_path
        self.recorded_start_date = recorded_start_date

    def convert(self, input: dict[int, dict[str, Any]]) -> FrameGroup:
        detection_parser = DetectionParser()
        frames = []
        for key, value in input.items():
            occurrence: datetime = datetime.strptime(
                str(value[OCCURRENCE]), DATE_FORMAT
            )
            data_detections = value[CLASSIFIED]
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
    time_without_frames: timedelta

    def __init__(self, no_frames_for: timedelta = timedelta(minutes=1)) -> None:
        self.time_without_frames = no_frames_for

    def run(self, files: list[Path]) -> PreprocessResult:
        input_data = {}
        for file in files:
            input = read_json(file)
            input_data[file] = input
        groups, metadata = self.process(input_data)
        return PreprocessResult(frame_groups=groups, metadata=metadata)

    def process(
        self, input: dict[Path, dict]
    ) -> Tuple[list[FrameGroup], dict[str, dict]]:
        all_groups, metadata = self._parse_frame_groups(input)
        if len(all_groups) == 0:
            return [], metadata
        return self._merge_groups(all_groups), metadata

    def _parse_frame_groups(
        self, input: dict[Path, dict]
    ) -> Tuple[list[FrameGroup], dict[str, dict]]:
        all_groups: list[FrameGroup] = []
        metadata: dict[str, dict] = {}
        for file_path, recording in input.items():
            file_metadata = recording[METADATA]
            metadata[file_path.as_posix()] = file_metadata
            start_date: datetime = self.extract_start_date_from(recording)
            data: dict[int, dict[str, Any]] = recording[DATA]
            frame_group = FrameGroupParser(
                file_path, recorded_start_date=start_date
            ).convert(data)
            all_groups.append(frame_group)
        return all_groups, metadata

    def _merge_groups(self, all_groups: list[FrameGroup]) -> list[FrameGroup]:
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
            return datetime.strptime(
                str(recording[METADATA][VIDEO][RECORDED_START_DATE]), DATE_FORMAT
            )
        return MISSING_START_DATE
