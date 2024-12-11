from datetime import datetime
from pathlib import Path
from typing import Any

from OTVision.dataformat import (
    CLASS,
    CONFIDENCE,
    DATA,
    DATE_FORMAT,
    DETECTIONS,
    OCCURRENCE,
    H,
    W,
    X,
    Y,
)
from OTVision.helpers.date import (
    parse_date_string_to_utc_datime,
    parse_timestamp_string_to_utc_datetime,
)
from OTVision.helpers.files import read_json
from OTVision.track.data import Detection, Frame
from OTVision.track.file_tracking import ChunkParser, FrameChunk


class JsonChunkParser(ChunkParser):

    def parse(self, file: Path, frame_offset: int = 0) -> FrameChunk:
        input: dict[int, dict[str, Any]] = read_json(file)[DATA]

        detection_parser = DetectionParser()
        frames = []
        for key, value in input.items():
            occurrence: datetime = parse_datetime(value[OCCURRENCE])
            data_detections = value[DETECTIONS]
            detections = detection_parser.convert(data_detections)
            parsed_frame = Frame(
                int(key) + frame_offset,
                occurrence=occurrence,
                source=file,
                detections=detections,
                image=None,
            )
            frames.append(parsed_frame)

        frames.sort(key=lambda frame: (frame.occurrence, frame.no))
        return FrameChunk(file, frames)


class DetectionParser:
    def convert(self, detection_data: list[dict[str, str]]) -> list[Detection]:
        detections: list[Detection] = []
        for detection in detection_data:
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
