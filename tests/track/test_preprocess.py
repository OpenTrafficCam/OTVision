from datetime import datetime, timedelta
from typing import Any

import pytest

from OTVision.track.preprocess import (
    CLASS,
    CLASSIFIED,
    CONFIDENCE,
    DATA,
    DATE_FORMAT,
    INPUT_FILE_PATH,
    OCCURRENCE,
    Cleanup,
    Detection,
    DetectionParser,
    Frame,
    FrameParser,
    H,
    W,
    X,
    Y,
)

DEFAULT_OCCURRENCE = datetime(year=2022, month=5, day=4)
DEFAULT_INPUT_FILE_PATH = "input-file.otdet"
DEFAULT_LABEL = "car"
DEFAULT_CONFIDENCE = 1.0
DEFAULT_X = 512.0
DEFAULT_Y = 256.0
DEFAULT_W = 128.0
DEFAULT_H = 64.0


def occurrence_from(key: int) -> datetime:
    return DEFAULT_OCCURRENCE + timedelta(microseconds=key)


def occurrence_as_string(key: int) -> str:
    return occurrence_from(key).strftime(DATE_FORMAT)


class DataBuilder:
    data: dict[str, dict[str, Any]]
    classified_frames: list[str]
    non_classified_frames: list[str]
    current_key: int

    def __init__(self) -> None:
        self.data = {}
        self.classified_frames = []
        self.non_classified_frames = []
        self.current_key = 0

    def append_non_classified_frame(
        self, input_file_path: str = DEFAULT_INPUT_FILE_PATH
    ) -> "DataBuilder":
        key: int = self.next_key()
        frame_number = str(key)
        occurrence = occurrence_from(key)
        self.data[frame_number] = {
            OCCURRENCE: occurrence,
            INPUT_FILE_PATH: input_file_path,
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
        input_file_path: str = DEFAULT_INPUT_FILE_PATH,
        label: str = DEFAULT_LABEL,
        confidence: float = DEFAULT_CONFIDENCE,
        x: float = DEFAULT_X,
        y: float = DEFAULT_Y,
        w: float = DEFAULT_W,
        h: float = DEFAULT_H,
    ) -> "DataBuilder":
        key: int = self.next_key()
        frame_number: str = str(key)
        occurrence = occurrence_as_string(key)
        self.data[frame_number] = {
            OCCURRENCE: occurrence,
            INPUT_FILE_PATH: input_file_path,
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
        input_file_path: str = DEFAULT_INPUT_FILE_PATH,
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
                input_file_path=input_file_path,
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

    def build(self) -> dict[str, dict[str, list]]:
        return self.data.copy()

    def build_as_detections(self) -> list[Frame]:
        parser = FrameParser()
        return parser.convert(self.data.copy())


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
        input: list[dict] = input_builder.append_classified_frame().build()["1"][
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

        parser = FrameParser()
        result: list[Frame] = parser.convert(input)

        assert result == [
            self.create_frame(1, [self.create_default_detection()]),
            self.create_frame(2, []),
            self.create_frame(3, [self.create_default_detection()]),
        ]

    def create_frame(self, frame_number: int, detections: list[Detection]) -> Frame:
        return Frame(
            frame=str(frame_number),
            occurrence=occurrence_from(frame_number),
            input_file_path=DEFAULT_INPUT_FILE_PATH,
            detections=detections,
        )

    def create_default_detection(self) -> Detection:
        return Detection(
            label=DEFAULT_LABEL,
            conf=DEFAULT_CONFIDENCE,
            x=DEFAULT_X,
            y=DEFAULT_Y,
            w=DEFAULT_W,
            h=DEFAULT_H,
        )


class TestPreprocess:
    def test_preprocess(self) -> None:
        builder = DataBuilder()
        builder.append_classified_frame(input_file_path="first-file.otdet")
