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
    H,
    W,
    X,
    Y,
)


class DataBuilder:
    DEFAULT_OCCURRENCE = datetime(year=2022, month=5, day=4)
    DEFAULT_INPUT_FILE_PATH = "input-file.otdet"
    DEFAULT_LABEL = "car"
    DEFAULT_CONFIDENCE = 1.0
    DEFAULT_X = 512.0
    DEFAULT_Y = 256.0
    DEFAULT_W = 128.0
    DEFAULT_H = 64.0

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
        occurrence = self.occurrence_from(key)
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
        occurrence = self.occurrence_as_string(key)
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

    def occurrence_as_string(self, key: int) -> str:
        return self.occurrence_from(key).strftime(DATE_FORMAT)

    def occurrence_from(self, key: int) -> datetime:
        return DataBuilder.DEFAULT_OCCURRENCE + timedelta(microseconds=key)

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

    def build_as_detections(self) -> list[Detection]:
        parser = DetectionParser()
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
        input = input_builder.append_classified_frame().build()

        parser = DetectionParser()
        result: list[Detection] = parser.convert(input)

        input_file_path: str = "input-file.otdet"
        expected_occurrence = input_builder.occurrence_from(1)
        assert result == [
            Detection(
                frame=1,
                occurrence=expected_occurrence,
                input_file_path=input_file_path,
                label=DataBuilder.DEFAULT_LABEL,
                conf=DataBuilder.DEFAULT_CONFIDENCE,
                x=DataBuilder.DEFAULT_X,
                y=DataBuilder.DEFAULT_Y,
                w=DataBuilder.DEFAULT_W,
                h=DataBuilder.DEFAULT_H,
            )
        ]


class TestPreprocess:
    def test_preprocess(self) -> None:
        builder = DataBuilder()
        builder.append_classified_frame(input_file_path="first-file.otdet")
