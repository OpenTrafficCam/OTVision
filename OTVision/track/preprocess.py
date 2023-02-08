from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from OTVision.helpers.files import get_files, read_json

METADATA: str = "metadata"
VIDEO: str = "vid"
FILE: str = "file"
RECORDED_START_DATE: str = "recorded_start_date"

DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S.%f"
INPUT_FILE_PATH: str = "input_file_path"
DATA: str = "data"
CLASS: str = "class"
CLASSIFIED: str = "classified"
FRAME: str = "frame"
OCCURRENCE: str = "occurrence"
LABEL: str = "label"
CONFIDENCE: str = "conf"
X: str = "x"
Y: str = "y"
W: str = "W"
H: str = "h"


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

    def to_dict(self) -> dict:
        return {
            LABEL: self.label,
            CONFIDENCE: self.conf,
            X: self.x,
            Y: self.y,
            W: self.w,
            H: self.h,
        }


@dataclass(frozen=True, repr=True)
class Frame:
    frame: int
    occurrence: datetime
    input_file_path: str
    detections: list[Detection]

    def to_dict(self) -> dict:
        return {
            FRAME: self.frame,
            OCCURRENCE: self.occurrence,
            INPUT_FILE_PATH: self.input_file_path,
            CLASSIFIED: self.detections,
        }


class Cleanup:
    def remove_empty_frames(
        self, data: dict[int, dict[str, list]]
    ) -> dict[int, dict[str, list]]:
        """
        Removes frames without detections from the given data object.

        Args:
            data (pd.DataFrame): data to remove empty frames from

        Returns:
            pd.DataFrame: same data object
        """
        keys_to_drop = []
        for key, value in data.items():
            classified_data = value[CLASSIFIED]
            if 0 == len(classified_data):
                keys_to_drop.append(key)
        [data.pop(key) for key in keys_to_drop]
        return data


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


class FrameParser:
    input_file_path: str
    recorded_start_date: datetime

    def __init__(self, input_file_path: str, recorded_start_date: datetime) -> None:
        self.input_file_path = input_file_path
        self.recorded_start_date = recorded_start_date

    def convert(self, input: dict[int, dict[str, Any]]) -> list[Frame]:
        detection_parser = DetectionParser()
        frames = []
        for key, value in input.items():
            occurrence: datetime = datetime.strptime(
                str(value[OCCURRENCE]), DATE_FORMAT
            )
            data_detections = value[CLASSIFIED]
            detections = detection_parser.convert(data_detections)
            parsed_frame = Frame(
                key,
                occurrence=occurrence,
                input_file_path=self.input_file_path,
                detections=detections,
            )
            frames.append(parsed_frame)
        return frames


class Preprocess:
    no_frames_for: timedelta

    def __init__(self, no_frames_for: timedelta) -> None:
        self.no_frames_for = no_frames_for

    def run(self, input_path: Path) -> None:
        input_data = []
        files = get_files([input_path], filetypes=[".otdet"])
        for file in files:
            input = read_json(file)
            input_data.append(input)
        self.process(input_data)

    def process(self, input: list[dict]) -> list[Frame]:
        all_detections = []
        for recording in input:
            input_file_path: str = str(recording[METADATA][VIDEO][FILE])
            start_date: datetime = self.extract_start_date_from(recording)
            data: dict[int, dict[str, Any]] = recording[DATA]
            detections = FrameParser(
                input_file_path, recorded_start_date=start_date
            ).convert(data)
            all_detections.extend(detections)
        return all_detections

    def extract_start_date_from(self, recording: dict) -> datetime:
        if RECORDED_START_DATE in recording[METADATA][VIDEO].keys():
            return datetime.strptime(
                str(recording[METADATA][VIDEO][RECORDED_START_DATE]), DATE_FORMAT
            )
        return datetime(1900, 1, 1)
