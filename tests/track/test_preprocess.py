from datetime import datetime, timedelta
from typing import Any

import pytest

from OTVision.track.preprocess import (
    CLASS,
    CLASSIFIED,
    CONFIDENCE,
    DATA,
    DATE_FORMAT,
    FILE,
    METADATA,
    OCCURRENCE,
    RECORDED_START_DATE,
    VIDEO,
    Cleanup,
    Detection,
    DetectionParser,
    Frame,
    FrameParser,
    H,
    Preprocess,
    W,
    X,
    Y,
)

DEFAULT_START_DATE = datetime(year=2022, month=5, day=4)
DEFAULT_INPUT_FILE_PATH = "input-file.otdet"
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


def occurrence_as_string(key: int, start_date: datetime = DEFAULT_START_DATE) -> str:
    return occurrence_from(key, start_date).strftime(DATE_FORMAT)


def create_frame(
    frame_number: int,
    detections: list[Detection],
    occurrence: datetime | None = None,
    input_file_path: str = DEFAULT_INPUT_FILE_PATH,
) -> Frame:
    default_occurrence = occurrence_from(frame_number)
    if occurrence is None:
        occurrence = default_occurrence
    return Frame(
        frame=frame_number,
        occurrence=occurrence,
        input_file_path=input_file_path,
        detections=detections,
    )


def create_default_detection() -> Detection:
    return Detection(
        label=DEFAULT_LABEL,
        conf=DEFAULT_CONFIDENCE,
        x=DEFAULT_X,
        y=DEFAULT_Y,
        w=DEFAULT_W,
        h=DEFAULT_H,
    )


class DataBuilder:
    data: dict[int, dict[str, Any]]
    classified_frames: list[int]
    non_classified_frames: list[int]
    current_key: int
    input_file_path: str
    start_date: datetime

    def __init__(
        self,
        input_file_path: str = DEFAULT_INPUT_FILE_PATH,
        start_date: datetime = DEFAULT_START_DATE,
    ) -> None:
        self.data = {}
        self.classified_frames = []
        self.non_classified_frames = []
        self.current_key = 0
        self.input_file_path = input_file_path
        self.start_date = start_date

    def append_non_classified_frame(self) -> "DataBuilder":
        frame_number = self.next_key()
        occurrence = occurrence_from(frame_number, start_date=self.start_date)
        self.data[frame_number] = {
            OCCURRENCE: occurrence,
            CLASSIFIED: [],
        }
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
    ) -> dict[str, object]:
        return {
            CLASS: label,
            CONFIDENCE: confidence,
            X: x,
            Y: y,
            W: w,
            H: h,
        }

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
        occurrence = occurrence_as_string(frame_number, self.start_date)
        self.data[frame_number] = {
            OCCURRENCE: occurrence,
            CLASSIFIED: [
                self.create_classification(
                    label=label,
                    confidence=confidence,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                )
                for i in range(0, number_of_classifications)
            ],
        }
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

    def build_as_detections(self) -> list[Frame]:
        parser = FrameParser(DEFAULT_INPUT_FILE_PATH, DEFAULT_START_DATE)
        return parser.convert(self.data.copy())

    def build_ot_det(self) -> dict:
        return {
            METADATA: {
                VIDEO: {
                    FILE: self.input_file_path,
                    RECORDED_START_DATE: self.start_date.strftime(DATE_FORMAT),
                }
            },
            DATA: self.build(),
        }


class TestCleanup:
    @pytest.mark.parametrize(
        DATA,
        [
            {},
            DataBuilder().append_classified_frame().build(),
        ],
    )
    def test_remove_nothing(self, data: dict) -> None:
        cleaned_data = Cleanup().remove_empty_frames(data.copy())

        assert cleaned_data == data

    def test_remove_frame_without_detection(self) -> None:
        data_builder = DataBuilder()
        data = (
            data_builder.append_non_classified_frame()
            .append_classified_frame()
            .append_non_classified_frame()
            .append_classified_frame()
            .append_classified_frame(2)
            .build()
        )

        cleaned_data = Cleanup().remove_empty_frames(data.copy())

        assert cleaned_data != data
        assert cleaned_data.keys() == set(data_builder.classified_frames)


class TestDetectionParser:
    detections: dict[str, list[Detection]]

    def test_convert(self) -> None:
        input_builder = DataBuilder()
        input: list[dict] = input_builder.append_classified_frame().build()[1][
            CLASSIFIED
        ]

        parser = DetectionParser()
        result: list[Detection] = parser.convert(input)

        assert result == [
            Detection(
                label=DEFAULT_LABEL,
                conf=DEFAULT_CONFIDENCE,
                x=DEFAULT_X,
                y=DEFAULT_Y,
                w=DEFAULT_W,
                h=DEFAULT_H,
            )
        ]


class TestFrameParser:
    frames: list[Frame]

    def test_convert(self) -> None:
        input_builder = DataBuilder()
        input_builder.append_classified_frame()
        input_builder.append_non_classified_frame()
        input_builder.append_classified_frame()
        input = input_builder.build()

        parser = FrameParser(
            DEFAULT_INPUT_FILE_PATH, recorded_start_date=DEFAULT_START_DATE
        )
        result: list[Frame] = parser.convert(input)

        expected_result = [
            create_frame(1, [create_default_detection()]),
            create_frame(2, []),
            create_frame(3, [create_default_detection()]),
        ]

        assert result == expected_result


class TestPreprocess:
    def test_preprocess(self) -> None:
        first_file_path = "first-file.otdet"
        first_start_date = datetime(2022, 5, 4, 12, 0, 0)
        first_builder = DataBuilder(
            input_file_path=first_file_path,
            start_date=first_start_date,
        )
        first_builder.append_classified_frame()
        first_detections = first_builder.build_ot_det()

        second_file_path = "second-file.otdet"
        second_start_date = datetime(2022, 5, 4, 12, 0, 1)
        second_builder = DataBuilder(
            input_file_path=second_file_path,
            start_date=second_start_date,
        )
        second_builder.append_classified_frame()
        second_detections = second_builder.build_ot_det()

        third_file_path = "third-file.otdet"
        third_start_date = datetime(2022, 5, 4, 13, 0, 1)
        third_builder = DataBuilder(
            input_file_path=third_file_path,
            start_date=third_start_date,
        )
        third_builder.append_classified_frame()
        third_detections = third_builder.build_ot_det()

        preprocessor = Preprocess(no_frames_for=timedelta(minutes=1))
        result = preprocessor.process(
            [first_detections, second_detections, third_detections]
        )

        expected_result = [
            Frame(
                1,
                input_file_path=first_file_path,
                occurrence=first_start_date,
                detections=[create_default_detection()],
            ),
            Frame(
                1,
                input_file_path=second_file_path,
                occurrence=second_start_date,
                detections=[create_default_detection()],
            ),
            Frame(
                1,
                input_file_path=third_file_path,
                occurrence=third_start_date,
                detections=[create_default_detection()],
            ),
        ]

        assert result == expected_result
