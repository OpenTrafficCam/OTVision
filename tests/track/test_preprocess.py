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
    TRACK_ID,
    VIDEO,
    H,
    W,
    X,
    Y,
)
from OTVision.track.preprocess import (
    Detection,
    DetectionParser,
    Frame,
    FrameChunk,
    FrameChunkParser,
    FrameGroup,
    FrameIndexer,
    Preprocess,
)

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
        occurrence = occurrence_serialized(frame_number, start_date=self.start_date)
        self.data[frame_number] = {
            FRAME: frame_number,
            OCCURRENCE: occurrence,
            INPUT_FILE_PATH: self.input_file_path.as_posix(),
            DETECTIONS: [],
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

    def build_as_detections(self) -> FrameChunk:
        return FrameChunkParser.convert(self.data.copy(), DEFAULT_INPUT_FILE_PATH)

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


class TestDetectionParser:
    detections: dict[str, list[Detection]]

    def test_convert(self) -> None:
        input_builder = DataBuilder()
        input: list[dict] = input_builder.append_classified_frame().build()[1][
            DETECTIONS
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


class TestFrame:
    def test_get_output_file(self) -> None:
        input_file = DEFAULT_INPUT_FILE_PATH
        frame = create_frame(1, [], input_file_path=input_file)

        suffix = ".suffix"
        output_file = frame.get_output_file(with_suffix=suffix)

        assert output_file == DEFAULT_INPUT_FILE_PATH.with_suffix(suffix=suffix)


class TestFrameParser:
    frames: list[Frame]

    def test_convert(self) -> None:
        input_builder = DataBuilder()
        input_builder.append_classified_frame()
        input_builder.append_non_classified_frame()
        input_builder.append_classified_frame()
        input = input_builder.build()

        order_key = "/some/path/to"
        path = Path(f"{order_key}/{DEFAULT_HOSTNAME}_2022-05-04_12-00-00.otdet")
        result: FrameChunk = FrameChunkParser.convert(input, path)

        expected_result = FrameChunk(
            path,
            [
                create_frame(1, [create_default_detection()], input_file_path=path),
                create_frame(2, [], input_file_path=path),
                create_frame(3, [create_default_detection()], input_file_path=path),
            ],
        )

        assert result == expected_result

    def test_convert_offset(self) -> None:
        input_builder = DataBuilder()
        input_builder.append_classified_frame()
        input_builder.append_non_classified_frame()
        input_builder.append_classified_frame()
        input = input_builder.build()

        order_key = "/some/path/to"
        path = Path(f"{order_key}/{DEFAULT_HOSTNAME}_2022-05-04_12-00-00.otdet")
        result: FrameChunk = FrameChunkParser.convert(input, path, frame_offset=5)

        expected_result = FrameChunk(
            path,
            [
                create_frame(
                    6,
                    [create_default_detection()],
                    occurrence=occurrence_from(1),
                    input_file_path=path,
                ),
                create_frame(
                    7, [], occurrence=occurrence_from(2), input_file_path=path
                ),
                create_frame(
                    8,
                    [create_default_detection()],
                    occurrence=occurrence_from(3),
                    input_file_path=path,
                ),
            ],
        )

        for f, g in zip(result.frames, expected_result.frames):
            print(f)
            print(g)
            print()

        assert result.file == expected_result.file
        assert result.frames[0] == expected_result.frames[0]
        assert result.frames[1] == expected_result.frames[1]
        assert result.frames[2] == expected_result.frames[2]


class TestPreprocess:
    def test_preprocess_single_file(self) -> None:
        order_key = "order-key"
        file_path = Path(f"{order_key}/{DEFAULT_HOSTNAME}_2022-05-04_12-00-01.otdet")
        start_date = datetime(2022, 5, 4, 12, 0, 1)
        builder = DataBuilder(
            input_file_path=file_path,
            start_date=start_date,
        )
        builder.append_classified_frame()
        otdet = builder.build_ot_det()

        preprocessor = Preprocess(time_without_frames=timedelta(minutes=1))
        preprocessed_otdet = preprocessor.process({Path(file_path): otdet[METADATA]})

        metadata = preprocessed_otdet[0]._files_metadata

        assert metadata == {
            file_path.as_posix(): {
                VIDEO: {
                    FILENAME: file_path.as_posix(),
                    RECORDED_START_DATE: start_date.timestamp(),
                    EXPECTED_DURATION: 1,
                }
            }
        }

    def test_preprocess_multiple_files(self) -> None:
        """
        https://openproject.platomo.de/projects/otcloud/work_packages/7527
        """
        order_key = "order-key"
        hostname = "first-host"
        first_file_path = Path(f"{order_key}/{hostname}_2022-05-04_12-00-01.otdet")
        first_start_date = datetime(2022, 5, 4, 12, 0, 1, tzinfo=timezone.utc)
        first_end_date = datetime(2022, 5, 4, 12, 0, 2, tzinfo=timezone.utc)
        first_builder = DataBuilder(
            input_file_path=first_file_path,
            start_date=first_start_date,
        )
        first_builder.append_classified_frame()
        first_detections = first_builder.build_ot_det()

        second_file_path = Path(f"{order_key}/{hostname}_2022-05-04_12-00-00.otdet")
        second_start_date = datetime(2022, 5, 4, 12, 0, 0, tzinfo=timezone.utc)
        second_builder = DataBuilder(
            input_file_path=second_file_path,
            start_date=second_start_date,
        )
        second_builder.append_classified_frame()
        second_detections = second_builder.build_ot_det()

        third_file_path = Path(f"{order_key}/{hostname}_non-hostname-part.otdet")
        third_start_date = datetime(2022, 5, 4, 13, 0, 1, tzinfo=timezone.utc)
        third_end_date = datetime(2022, 5, 4, 13, 0, 2, tzinfo=timezone.utc)
        third_builder = DataBuilder(
            input_file_path=third_file_path,
            start_date=third_start_date,
        )
        third_builder.append_classified_frame()
        third_detections = third_builder.build_ot_det()

        preprocessor = Preprocess(time_without_frames=timedelta(minutes=1))
        merged_groups = preprocessor.process(
            {
                first_file_path: first_detections[METADATA],
                second_file_path: second_detections[METADATA],
                third_file_path: third_detections[METADATA],
            }
        )

        assert len(merged_groups) == 2
        assert merged_groups[0].start_date() == second_start_date
        assert merged_groups[0].end_date() == first_end_date
        assert merged_groups[0].hostname == hostname
        assert merged_groups[1].start_date() == third_start_date
        assert merged_groups[1].end_date() == third_end_date
        assert merged_groups[1].hostname == hostname

        assert merged_groups[0]._files_metadata == {
            first_file_path.as_posix(): {
                VIDEO: {
                    FILENAME: first_file_path.as_posix(),
                    RECORDED_START_DATE: first_start_date.timestamp(),
                    EXPECTED_DURATION: 1,
                }
            },
            second_file_path.as_posix(): {
                VIDEO: {
                    FILENAME: second_file_path.as_posix(),
                    RECORDED_START_DATE: second_start_date.timestamp(),
                    EXPECTED_DURATION: 1,
                }
            },
        }
        assert merged_groups[1]._files_metadata == {
            third_file_path.as_posix(): {
                VIDEO: {
                    FILENAME: third_file_path.as_posix(),
                    RECORDED_START_DATE: third_start_date.timestamp(),
                    EXPECTED_DURATION: 1,
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
        assert merge_first_second.hostname == DEFAULT_HOSTNAME
        assert merge_first_second.files == merge_second_first.files
        assert merge_first_second.start_date() == merge_second_first.start_date()
        assert merge_first_second.end_date() == merge_second_first.end_date()
        assert merge_first_second.hostname == merge_second_first.hostname

    def _create_frame_group(
        self,
        start_date: datetime = DEFAULT_START_DATE,
        end_date: datetime = DEFAULT_START_DATE,
        hostname: str = DEFAULT_HOSTNAME,
        input_file_path: Path = DEFAULT_INPUT_FILE_PATH,
    ) -> FrameGroup:
        return FrameGroup(
            start_date=start_date,
            end_date=end_date,
            hostname=hostname,
            file=input_file_path,
            metadata={
                VIDEO: {
                    FILENAME: input_file_path.as_posix(),
                    RECORDED_START_DATE: start_date.timestamp(),
                    EXPECTED_DURATION: 1,
                }
            },
        )


class TestSplitter:
    def test_split(self) -> None:
        input_builder = DataBuilder()
        input_builder.append_classified_frame()
        input_builder.append_non_classified_frame()
        input_builder.append_classified_frame()

        expected_result, tracked_frames = self.create_test_data(input_builder)

        indexer = FrameIndexer()
        result = indexer.reindex(tracked_frames, frame_offset=0)

        assert result == expected_result

    @staticmethod
    def create_test_data(
        input_builder: DataBuilder,
        first_frame: int = 1,
        offset: int = 0,
    ) -> tuple[list, dict]:
        frames = input_builder.build_as_detections().to_dict()
        track_id = "1"
        tracked_frames: dict[str, dict] = {
            str(first_frame): {
                key: value | {TRACK_ID: track_id} for key, value in frames[DATA].items()
            }
        }
        expected_result = [
            value | {TRACK_ID: track_id, FRAME: value[FRAME] - offset}
            for key, value in frames[DATA].items()
        ]

        return expected_result, tracked_frames

    def test_split_with_empty_frames_at_the_beginning(self) -> None:
        input_builder = DataBuilder()
        input_builder.next_key()
        input_builder.append_classified_frame()
        frame_offset = 2

        expected_result, tracked_frames = self.create_test_data(
            input_builder, first_frame=3, offset=frame_offset
        )

        indexer = FrameIndexer()
        result = indexer.reindex(tracked_frames, frame_offset)

        assert result == expected_result

    def test_split_empty(self) -> None:
        tracked_frames: dict[str, dict] = {}
        expected_result: list[dict] = []

        indexer = FrameIndexer()
        result = indexer.reindex(tracked_frames, frame_offset=0)

        assert result == expected_result
