from dataclasses import dataclass
from pathlib import Path

from OTVision.helpers.files import read_json

DATA: str = "data"
CLASS: str = "class"
CLASSIFIED: str = "classified"
FRAME: str = "frame"
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

    frame: int
    label: str
    conf: float
    x: float
    y: float
    w: float
    h: float

    def to_dict(self) -> dict:
        return {
            FRAME: self.frame,
            LABEL: self.label,
            CONFIDENCE: self.conf,
            X: self.x,
            Y: self.y,
            W: self.w,
            H: self.h,
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
    def convert(self, input: dict[str, dict[str, list]]) -> list[Detection]:
        detections: list[Detection] = []
        for key, value in input.items():
            data_detections = value[CLASSIFIED]
            for detection in data_detections:
                detected_item = Detection(
                    int(key),
                    detection[CLASS],
                    detection[CONFIDENCE],
                    detection[X],
                    detection[Y],
                    detection[W],
                    detection[H],
                )
                detections.append(detected_item)
        return detections


class Preprocess:
    def load_data(self, input_path: Path) -> list[Detection]:
        input = read_json(input_path)
        data: dict[str, dict[str, list]] = input[DATA]
        cleaned = Cleanup().remove_empty_frames(data)
        return DetectionParser().convert(cleaned)
