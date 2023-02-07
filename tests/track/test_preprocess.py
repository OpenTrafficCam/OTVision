import pytest

from track.preprocess import Cleanup, Detection, DetectionParser


class DataBuilder:
    DEFAULT_LABEL = "car"
    DEFAULT_CONFIDENCE = 1.0
    DEFAULT_X = 512.0
    DEFAULT_Y = 256.0
    DEFAULT_W = 128.0
    DEFAULT_H = 64.0

    data: dict[str, dict[str, list]]
    classified_frames: list[str]
    non_classified_frames: list[str]
    current_key: int

    def __init__(self) -> None:
        self.data = {}
        self.classified_frames = []
        self.non_classified_frames = []
        self.current_key = 0

    def append_non_classified_frame(self) -> "DataBuilder":
        frame_number = self.next_key()
        self.data[frame_number] = {"classified": []}
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
            "class": label,
            "conf": confidence,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
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
        frame_number = self.next_key()
        self.data[frame_number] = {
            "classified": [
                self.create_classification(
                    label=label, confidence=confidence, x=x, y=y, w=w, h=h
                )
                for i in range(0, number_of_classifications)
            ]
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
                number_of_classifications, label, confidence, x, y, w, h
            )
        return self

    def next_key(self) -> str:
        self.current_key += 1
        return str(self.current_key)

    def build(self) -> dict[str, dict[str, list]]:
        return self.data.copy()

    def build_as_detections(self) -> list[Detection]:
        parser = DetectionParser()
        return parser.convert(self.data.copy())


class TestCleanup:
    @pytest.mark.parametrize(
        "data",
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

        assert result == [
            Detection(
                frame=1,
                label=DataBuilder.DEFAULT_LABEL,
                conf=DataBuilder.DEFAULT_CONFIDENCE,
                x=DataBuilder.DEFAULT_X,
                y=DataBuilder.DEFAULT_Y,
                w=DataBuilder.DEFAULT_W,
                h=DataBuilder.DEFAULT_H,
            )
        ]
