from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pytest

from OTVision.dataformat import (
    CLASS,
    CLASSIFIED,
    CONFIDENCE,
    DATA,
    DATE_FORMAT,
    FILE,
    FRAME,
    INPUT_FILE_PATH,
    METADATA,
    OCCURRENCE,
    RECORDED_START_DATE,
    VIDEO,
    H,
    W,
    X,
    Y,
)
from OTVision.track.preprocess import (
    Cleanup,
    Detection,
    DetectionParser,
    Frame,
    FrameGroup,
    FrameGroupParser,
    Preprocess,
)

DEFAULT_START_DATE = datetime(year=2022, month=5, day=4)
DEFAULT_INPUT_FILE_PATH = Path("input-file.otdet")
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
    occurrence: Optional[datetime] = None,
    input_file_path: Path = DEFAULT_INPUT_FILE_PATH,
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
    input_file_path: Path
    start_date: datetime

    def __init__(
        self,
        input_file_path: Path = DEFAULT_INPUT_FILE_PATH,
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
            FRAME: frame_number,
            OCCURRENCE: occurrence,
            INPUT_FILE_PATH: self.input_file_path.as_posix(),
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
        frame_number: int = 1,
        occurrence: str = DEFAULT_START_DATE.strftime(DATE_FORMAT),
        input_file_path: Path = DEFAULT_INPUT_FILE_PATH,
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
            FRAME: frame_number,
            OCCURRENCE: occurrence,
            INPUT_FILE_PATH: self.input_file_path.as_posix(),
            CLASSIFIED: [
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

    def build_as_detections(self) -> FrameGroup:
        parser = FrameGroupParser(DEFAULT_INPUT_FILE_PATH, DEFAULT_START_DATE)
        return parser.convert(self.data.copy())

    def build_ot_det(self) -> dict:
        return {
            METADATA: {
                VIDEO: {
                    FILE: self.input_file_path.as_posix(),
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

        order_key = "/some/path/to"
        path = Path(f"{order_key}/file-name.otdet")
        parser = FrameGroupParser(path, recorded_start_date=DEFAULT_START_DATE)
        result: FrameGroup = parser.convert(input)

        expected_result = FrameGroup(
            [
                create_frame(1, [create_default_detection()], input_file_path=path),
                create_frame(2, [], input_file_path=path),
                create_frame(3, [create_default_detection()], input_file_path=path),
            ],
            order_key=order_key,
        )

        assert result == expected_result

    def test_order_key(self) -> None:
        order_key = "/some/path/to"
        path = Path(f"{order_key}/file-name.otdet")
        parser = FrameGroupParser(path, recorded_start_date=DEFAULT_START_DATE)

        calculated_key = parser.order_key()

        assert calculated_key == order_key


class TestPreprocess:
    def test_preprocess_single_file(self) -> None:
        order_key = "order-key"
        file_path = Path(f"{order_key}/first-file.otdet")
        start_date = datetime(2022, 5, 4, 12, 0, 1)
        builder = DataBuilder(
            input_file_path=file_path,
            start_date=start_date,
        )
        builder.append_classified_frame()
        otdet = builder.build_ot_det()

        preprocessor = Preprocess(no_frames_for=timedelta(minutes=1))
        preprocessed_otdet, metadata = preprocessor.process({Path(file_path): otdet})
        serialized_otdet = preprocessed_otdet[0].to_dict()

        assert serialized_otdet == {DATA: otdet[DATA]}
        assert metadata == {
            file_path.as_posix(): {
                VIDEO: {
                    FILE: file_path.as_posix(),
                    RECORDED_START_DATE: start_date.strftime(DATE_FORMAT),
                }
            }
        }

    def test_preprocess_multiple_files(self) -> None:
        order_key = "order-key"
        first_file_path = Path(f"{order_key}/first-file.otdet")
        first_start_date = datetime(2022, 5, 4, 12, 0, 1)
        first_builder = DataBuilder(
            input_file_path=first_file_path,
            start_date=first_start_date,
        )
        first_builder.append_classified_frame()
        first_detections = first_builder.build_ot_det()

        second_file_path = Path(f"{order_key}/second-file.otdet")
        second_start_date = datetime(2022, 5, 4, 12, 0, 0)
        second_builder = DataBuilder(
            input_file_path=second_file_path,
            start_date=second_start_date,
        )
        second_builder.append_classified_frame()
        second_detections = second_builder.build_ot_det()

        third_file_path = Path(f"{order_key}/third-file.otdet")
        third_start_date = datetime(2022, 5, 4, 13, 0, 1)
        third_builder = DataBuilder(
            input_file_path=third_file_path,
            start_date=third_start_date,
        )
        third_builder.append_classified_frame()
        third_detections = third_builder.build_ot_det()

        preprocessor = Preprocess(no_frames_for=timedelta(minutes=1))
        merged_groups, metadata = preprocessor.process(
            {
                first_file_path: first_detections,
                second_file_path: second_detections,
                third_file_path: third_detections,
            }
        )

        expected_result = [
            FrameGroup(
                [
                    Frame(
                        1,
                        occurrence=second_start_date,
                        input_file_path=second_file_path,
                        detections=[create_default_detection()],
                    ),
                    Frame(
                        2,
                        occurrence=first_start_date,
                        input_file_path=first_file_path,
                        detections=[create_default_detection()],
                    ),
                ],
                order_key=order_key,
            ),
            FrameGroup(
                [
                    Frame(
                        1,
                        occurrence=third_start_date,
                        input_file_path=third_file_path,
                        detections=[create_default_detection()],
                    )
                ],
                order_key=order_key,
            ),
        ]

        assert merged_groups == expected_result
        assert metadata == {
            first_file_path.as_posix(): {
                VIDEO: {
                    FILE: first_file_path.as_posix(),
                    RECORDED_START_DATE: first_start_date.strftime(DATE_FORMAT),
                }
            },
            second_file_path.as_posix(): {
                VIDEO: {
                    FILE: second_file_path.as_posix(),
                    RECORDED_START_DATE: second_start_date.strftime(DATE_FORMAT),
                }
            },
            third_file_path.as_posix(): {
                VIDEO: {
                    FILE: third_file_path.as_posix(),
                    RECORDED_START_DATE: third_start_date.strftime(DATE_FORMAT),
                }
            },
        }


class TestFrameGroup:
    def test_start_date(self) -> None:
        start_date = datetime(2022, 5, 4, 12, 0, 0)
        group = self._create_frame_group(start_date=start_date)

        assert group.start_date() == start_date

    def test_end_date(self) -> None:
        end_date = datetime(2022, 5, 4, 12, 0, 1)
        group = self._create_frame_group(end_date=end_date)

        assert group.end_date() == end_date

    def test_merge(self) -> None:
        first_start = datetime(2022, 5, 4, 12, 0, 0)
        first_end = first_start + timedelta(seconds=1)
        second_start = first_end + timedelta(seconds=1)
        second_end = second_start + timedelta(seconds=1)
        first_group = self._create_frame_group(
            start_date=first_start, end_date=first_end
        )
        second_group = self._create_frame_group(
            start_date=second_end, end_date=second_end
        )

        merge_first_second: FrameGroup = first_group.merge(second_group)
        merge_second_first: FrameGroup = second_group.merge(first_group)

        assert merge_first_second.start_date() == first_start
        assert merge_first_second.end_date() == second_end
        assert merge_first_second == merge_second_first

    def _create_frame_group(
        self,
        order_key: str = "/some/path/to",
        start_date: datetime = DEFAULT_START_DATE,
        end_date: datetime = DEFAULT_START_DATE,
        input_file_path: Path = DEFAULT_INPUT_FILE_PATH,
    ) -> FrameGroup:
        frames: list[Frame] = [
            create_frame(1, [], occurrence=start_date, input_file_path=input_file_path),
            create_frame(2, [], occurrence=end_date, input_file_path=input_file_path),
        ]
        path = f"{order_key}/detection.otdet"
        group = FrameGroup(frames, order_key=path)
        return group
