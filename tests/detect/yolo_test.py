from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from cv2 import VideoCapture
from torch import Tensor

from OTVision.detect.yolo import Yolov8
from OTVision.track.preprocess import Detection


@pytest.fixture
def video_path() -> str:
    return str(
        Path(__file__).parents[1]
        / "data"
        / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"
    )


@pytest.fixture
def num_frames(video_path: str) -> int:
    cap = VideoCapture(video_path)

    len_frames = 0

    while True:
        gotframe, _ = cap.read()
        if not gotframe:
            break
        len_frames += 1
    cap.release()

    return len_frames


class TestConvertDetections:
    def test_convert_x_y_coordinates(self) -> None:
        classification: int = 1
        name: str = "name"
        names = {classification: name}
        x_input = 20
        x_output = 15
        y_input = 20
        y_output = 15
        width = 10
        height = 10
        confidence = 0.5

        mock_yolo = Mock().return_value
        mock_yolo.names = names

        model = Yolov8(
            weights=Mock(),
            model=mock_yolo,
            confidence=0.25,
            iou=0.25,
            img_size=640,
            half_precision=False,
            normalized=False,
            frame_rotator=Mock(),
        )

        result = model._parse_detection(
            Tensor([x_input, y_input, width, height]), classification, confidence
        )

        assert result == Detection(name, confidence, x_output, y_output, width, height)


class TestObjectDetection:
    @patch("OTVision.detect.yolo.Yolov8._parse_detections")
    @patch("OTVision.detect.yolo.av")
    def test_detection_start_and_end_are_considered(
        self, mock_av: Mock, mock_parse_detections: Mock
    ) -> None:
        detect_start = 300
        detect_end = 600
        total_frames = 900
        file = Path("path/to/video.mp4")
        frame_rotator = Mock()
        yolo_model = Mock()
        container = MagicMock()
        context_manager_container = MagicMock()
        video_frames = self._create_mock_frames(total_frames)
        rotated_frame = Mock()
        predicted_frame = Mock()
        parsed_detection = Mock()

        mock_av.open.return_value = container
        container.__enter__.return_value = context_manager_container
        context_manager_container.decode.return_value = video_frames
        frame_rotator.rotate.return_value = rotated_frame
        yolo_model.predict.return_value = [predicted_frame]
        mock_parse_detections.return_value = parsed_detection
        get_number_of_frames = Mock(return_value=total_frames)

        target = Yolov8(
            weights=Mock(),
            model=yolo_model,
            confidence=0.25,
            iou=0.25,
            img_size=640,
            half_precision=False,
            normalized=False,
            frame_rotator=frame_rotator,
            get_number_of_frames=get_number_of_frames,
        )
        actual = target.detect(file, detect_start, detect_end)

        mock_av.open.assert_called_once_with(str(file.absolute()))
        context_manager_container.decode.assert_called_once_with(video=0)
        self.assert_detections_are_correct(
            actual, parsed_detection, detect_start, detect_end
        )
        assert frame_rotator.rotate.call_count == 300
        assert yolo_model.predict.call_count == 300
        assert mock_parse_detections.call_count == 300

    def assert_detections_are_correct(
        self,
        detections: list[list],
        expected: Mock,
        detect_start: int,
        detect_end: int,
    ) -> None:
        for i, actual in enumerate(detections, start=1):
            if detect_start <= i < detect_end:
                assert actual == expected
            else:
                assert actual == []

    def _create_mock_frames(self, total_frames: int) -> list[Mock]:
        image = Mock()
        return [image for _ in range(total_frames)]
