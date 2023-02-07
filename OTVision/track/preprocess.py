from dataclasses import dataclass

from helpers.files import read_json


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
            "frame": self.frame,
            "label": self.label,
            "conf": self.conf,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
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
            classified_data = value["classified"]
            if 0 == len(classified_data):
                keys_to_drop.append(key)
        [data.pop(key) for key in keys_to_drop]
        return data


class DetectionParser:
    def convert(self, input: dict[str, dict[str, list]]) -> list[Detection]:
        detections: list[Detection] = []
        for key, value in input.items():
            data_detections = value["classified"]
            for detection in data_detections:
                detected_item = Detection(
                    int(key),
                    detection["class"],
                    detection["conf"],
                    detection["x"],
                    detection["y"],
                    detection["w"],
                    detection["h"],
                )
                detections.append(detected_item)
        return detections


class Preprocess:
    def load_data(self, input_path: str) -> list[Detection]:
        input = read_json(input_path)
        data: dict[str, dict[str, list]] = input["data"]
        cleaned = Cleanup().remove_empty_frames(data)
        return DetectionParser().convert(cleaned)
