from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

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
from OTVision.domain.detection import Detection
from OTVision.domain.frame import DetectedFrame
from OTVision.helpers.date import (
    parse_date_string_to_utc_datime,
    parse_timestamp_string_to_utc_datetime,
)
from OTVision.helpers.files import denormalize_bbox, read_json
from OTVision.track.model.filebased.frame_chunk import ChunkParser, FrameChunk
from OTVision.track.model.filebased.frame_group import FrameGroup


class JsonChunkParser(ChunkParser):

    def parse(
        self, file: Path, frame_group: FrameGroup, frame_offset: int = 0
    ) -> FrameChunk:
        json = read_json(file)
        metadata: dict = frame_group.metadata_by_file[file]

        denormalized = denormalize_bbox(
            json, file, metadata={file.as_posix(): metadata}
        )
        input: dict[int, dict[str, Any]] = denormalized[DATA]

        frames = self.convert(file, frame_offset, input)

        frames.sort(key=lambda frame: (frame.occurrence, frame.no))
        return FrameChunk(file, metadata, frames, frame_group.id)

    def convert(
        self, file: Path, frame_offset: int, input: dict[int, dict[str, Any]]
    ) -> list[DetectedFrame]:
        detection_parser = DetectionParser()
        frames = []

        input_progress = tqdm(
            input.items(), desc="parse Frames", total=len(input), leave=False
        )
        for key, value in input_progress:
            occurrence: datetime = parse_datetime(value[OCCURRENCE])
            data_detections = value[DETECTIONS]
            detections = detection_parser.convert(data_detections)
            parsed_frame = DetectedFrame(
                no=int(key) + frame_offset,
                occurrence=occurrence,
                source=str(file),
                output=str(file),
                detections=detections,
                image=None,
            )
            frames.append(parsed_frame)
        return frames


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
