from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from OTVision.helpers.files import read_json

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
    frame: str
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
        self, data: dict[str, dict[str, list]]
    ) -> dict[str, dict[str, list]]:
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
    def convert(self, input: dict[str, dict[str, Any]]) -> list[Frame]:
        detection_parser = DetectionParser()
        frames = []
        for key, value in input.items():
            occurrence: datetime = datetime.strptime(
                str(value[OCCURRENCE]), DATE_FORMAT
            )
            input_file_path: str = str(value[INPUT_FILE_PATH])
            data_detections = value[CLASSIFIED]
            detections = detection_parser.convert(data_detections)
            parsed_frame = Frame(
                key,
                occurrence=occurrence,
                input_file_path=input_file_path,
                detections=detections,
            )
            frames.append(parsed_frame)
        return frames


class Preprocess:
    def load_data(self, input_path: Path) -> list[Frame]:
        input = read_json(input_path)
        data: dict[str, dict[str, Any]] = input[DATA]
        return FrameParser().convert(data)
