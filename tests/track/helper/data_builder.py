from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from OTVision.dataformat import (
    CLASS,
    CONFIDENCE,
    DATA,
    DETECTIONS,
    EXPECTED_DURATION,
    FILENAME,
    FRAME,
    INPUT_FILE_PATH,
    INTERPOLATED_DETECTION,
    METADATA,
    OCCURRENCE,
    RECORDED_START_DATE,
    VIDEO,
    H,
    W,
    X,
    Y,
)
from OTVision.domain.detection import Detection
from OTVision.domain.frame import DetectedFrame
from OTVision.track.parser.chunk_parser_plugins import JsonChunkParser

DEFAULT_HOSTNAME = "hostname"
DEFAULT_START_DATE = datetime(year=2022, month=5, day=4, tzinfo=timezone.utc)
DEFAULT_INPUT_FILE_PATH = Path(f"{DEFAULT_HOSTNAME}_input-file.otdet")
DEFAULT_LABEL = "car"
DEFAULT_CONFIDENCE = 1.0
DEFAULT_X = 512.0
DEFAULT_Y = 256.0
DEFAULT_W = 128.0
DEFAULT_H = 64.0


def occurrence_from(key: int, start_date: datetime = DEFAULT_START_DATE) -> datetime:
    if start_date == DEFAULT_START_DATE:
        return start_date + timedelta(microseconds=key)
    return start_date


def occurrence_serialized(key: int, start_date: datetime = DEFAULT_START_DATE) -> float:
    return occurrence_from(key, start_date).timestamp()


class DataBuilder:
    data: dict[int, dict[str, Any]]
    objects: dict[int, DetectedFrame]
    classified_frames: list[int]
    non_classified_frames: list[int]
    current_key: int
    input_file_path: Path
    start_date: datetime

    def __init__(
        self,
        input_file_path: Path = DEFAULT_INPUT_FILE_PATH,
        start_date: datetime = DEFAULT_START_DATE,
    ) -> None:
        self.data = {}
        self.objects = {}
        self.classified_frames = []
        self.non_classified_frames = []
        self.current_key = 0
        self.input_file_path = input_file_path
        self.start_date = start_date

    def append_non_classified_frame(self) -> "DataBuilder":
        frame_number = self.next_key()
        occurrence = occurrence_serialized(frame_number, start_date=self.start_date)
        self.data[frame_number] = {
            FRAME: frame_number,
            OCCURRENCE: occurrence,
            INPUT_FILE_PATH: self.input_file_path.as_posix(),
            DETECTIONS: [],
        }

        occurrence_date = occurrence_from(frame_number, self.start_date)
        self.objects[frame_number] = DetectedFrame(
            no=frame_number,
            occurrence=occurrence_date,
            source=str(self.input_file_path),
            output=str(self.input_file_path),
            detections=[],
            image=None,
        )
        self.non_classified_frames.append(frame_number)
        return self

    def batch_append_non_classified_frame(
        self, number_of_frames: int = 1
    ) -> "DataBuilder":
        for i in range(0, number_of_frames):
            self.append_non_classified_frame()
        return self

    def create_classification(
        self,
        label: str = DEFAULT_LABEL,
        confidence: float = DEFAULT_CONFIDENCE,
        x: float = DEFAULT_X,
        y: float = DEFAULT_Y,
        w: float = DEFAULT_W,
        h: float = DEFAULT_H,
        frame_number: int = 1,
        occurrence: float = DEFAULT_START_DATE.timestamp(),
        input_file_path: Path = DEFAULT_INPUT_FILE_PATH,
        interpolated_detection: bool = False,
    ) -> dict[str, object]:
        return {
            CLASS: label,
            CONFIDENCE: confidence,
            X: x,
            Y: y,
            W: w,
            H: h,
            FRAME: frame_number,
            OCCURRENCE: occurrence,
            INPUT_FILE_PATH: input_file_path.as_posix(),
            INTERPOLATED_DETECTION: interpolated_detection,
        }

    def create_classification_object(
        self,
        label: str = DEFAULT_LABEL,
        confidence: float = DEFAULT_CONFIDENCE,
        x: float = DEFAULT_X,
        y: float = DEFAULT_Y,
        w: float = DEFAULT_W,
        h: float = DEFAULT_H,
    ) -> Detection:
        return Detection(label=label, conf=confidence, x=x, y=y, w=w, h=h)

    def append_classified_frame(
        self,
        number_of_classifications: int = 1,
        label: str = DEFAULT_LABEL,
        confidence: float = DEFAULT_CONFIDENCE,
        x: float = DEFAULT_X,
        y: float = DEFAULT_Y,
        w: float = DEFAULT_W,
        h: float = DEFAULT_H,
    ) -> "DataBuilder":
        frame_number: int = self.next_key()
        occurrence = occurrence_serialized(frame_number, self.start_date)
        self.data[frame_number] = {
            FRAME: frame_number,
            OCCURRENCE: occurrence,
            INPUT_FILE_PATH: self.input_file_path.as_posix(),
            DETECTIONS: [
                self.create_classification(
                    label=label,
                    confidence=confidence,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    frame_number=frame_number,
                    occurrence=occurrence,
                    input_file_path=self.input_file_path,
                )
                for i in range(0, number_of_classifications)
            ],
        }

        occurrence_date = occurrence_from(frame_number, self.start_date)
        self.objects[frame_number] = DetectedFrame(
            no=frame_number,
            occurrence=occurrence_date,
            source=str(self.input_file_path),
            output=str(self.input_file_path),
            detections=[
                self.create_classification_object(label, confidence, x, y, w, h)
                for i in range(0, number_of_classifications)
            ],
            image=None,
        )
        self.classified_frames.append(frame_number)
        return self

    def batch_append_classified_frames(
        self,
        number_of_frames: int = 1,
        number_of_classifications: int = 1,
        label: str = DEFAULT_LABEL,
        confidence: float = DEFAULT_CONFIDENCE,
        x: float = DEFAULT_X,
        y: float = DEFAULT_Y,
        w: float = DEFAULT_W,
        h: float = DEFAULT_H,
    ) -> "DataBuilder":
        for i in range(0, number_of_frames):
            self.append_classified_frame(
                number_of_classifications=number_of_classifications,
                label=label,
                confidence=confidence,
                x=x,
                y=y,
                w=w,
                h=h,
            )
        return self

    def next_key(self) -> int:
        self.current_key += 1
        return self.current_key

    def build(self) -> dict[int, dict[str, list]]:
        return self.data.copy()

    def build_objects(self) -> dict[int, DetectedFrame]:
        return self.objects.copy()

    def build_as_detections(self) -> list[DetectedFrame]:
        return JsonChunkParser().convert(DEFAULT_INPUT_FILE_PATH, 0, self.data.copy())

    def build_ot_det(self) -> dict:
        return {
            METADATA: {
                VIDEO: {
                    FILENAME: self.input_file_path.as_posix(),
                    RECORDED_START_DATE: self.start_date.timestamp(),
                    EXPECTED_DURATION: 1,
                }
            },
            DATA: self.build(),
        }


def create_default_detection() -> Detection:
    return Detection(
        label=DEFAULT_LABEL,
        conf=DEFAULT_CONFIDENCE,
        x=DEFAULT_X,
        y=DEFAULT_Y,
        w=DEFAULT_W,
        h=DEFAULT_H,
    )


def create_frame(
    frame_number: int,
    detections: list[Detection],
    occurrence: Optional[datetime] = None,
    input_file_path: Path = DEFAULT_INPUT_FILE_PATH,
) -> DetectedFrame:
    default_occurrence = occurrence_from(frame_number)
    if occurrence is None:
        occurrence = default_occurrence

    return DetectedFrame(
        no=frame_number,
        occurrence=occurrence,
        source=str(input_file_path),
        output=str(input_file_path),
        detections=detections,
        image=None,
    )
