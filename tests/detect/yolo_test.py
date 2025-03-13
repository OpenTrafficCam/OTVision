from typing import Any
from unittest.mock import Mock, call, patch

import pytest
from torch import Tensor
from ultralytics.engine.results import Boxes, Results

from OTVision.config import Config, DetectConfig
from OTVision.detect.yolo import YoloDetectionConverter, YoloDetector
from OTVision.domain.detection import DetectedFrame, Detection
from OTVision.domain.frame import Frame, FrameKeys
from tests.utils.generator import make_generator
from tests.utils.mocking import create_mocks

FPS = 20
SOURCE = "path/to/video.mp4"


class TestYoloDetectionConverter:
    @pytest.mark.parametrize("normalized", [True, False])
    def test_convert(self, normalized: bool) -> None:
        given_class_mapping = {1: "car"}
        given_boxes = self.create_boxes()

        target = YoloDetectionConverter()

        actual = target.convert(given_boxes, normalized, given_class_mapping)

        assert actual == self.expected_detections(
            given_boxes, normalized, given_class_mapping
        )

    def expected_detections(
        self, boxes: Boxes, normalized: bool, class_mapping: dict[int, str]
    ) -> list[Detection]:

        if normalized:
            x, y, w, h = boxes.xywhn[0].tolist()
        else:
            x, y, w, h = boxes.xywh[0].tolist()

        return [
            Detection(
                label=class_mapping[int(boxes.cls.item())],
                conf=boxes.conf.item(),
                x=x - w / 2,
                y=y - w / 2,
                w=w,
                h=h,
            )
        ]

    def create_boxes(self) -> Boxes:
        return Boxes(Tensor([10, 10, 20, 20, 0.5, 1]), orig_shape=(100, 100))


class TestYoloDetector:
    @patch("OTVision.detect.yolo.torch")
    def test_detect(self, mock_torch: Mock) -> None:
        mock_torch.cuda.is_available.return_value = False
        expected_model_predictions: list[Results] = create_mocks(2)
        expected_detections: list[Detection] = create_mocks(2)
        expected_detected_frames: list[DetectedFrame] = create_mocks(4)

        config = Config()
        given_model = self.create_model(expected_model_predictions)
        given_current_config = self.create_get_current_config(config)
        given_detection_converter = self.create_detection_converter(expected_detections)
        given_detected_frame_factory = self.create_detected_frame_factory(
            expected_detected_frames
        )

        target = YoloDetector(
            model=given_model,
            get_current_config=given_current_config,
            detection_converter=given_detection_converter,
            detected_frame_factory=given_detected_frame_factory,
        )
        given_input_frames = self.creat_input_frames()
        actual = list(target.detect(make_generator(given_input_frames)))

        assert actual == expected_detected_frames
        self.assert_model_called(given_model, given_input_frames, config.detect)
        self.assert_detected_frame_factory_called(
            given_detected_frame_factory, given_input_frames, expected_detections
        )
        self.assert_detection_converter_called(
            given_detection_converter,
            expected_model_predictions,
            normalized=config.detect.normalized,
            class_mapping=target.classifications,
        )

    def assert_model_called(
        self, model: Mock, input_frames: list[Frame], config: DetectConfig
    ) -> None:
        def create_call(source: Any) -> Any:
            return call(
                source=source,
                conf=config.confidence,
                iou=config.iou,
                half=config.half_precision,
                imgsz=config.img_size,
                device="cpu",
                stream=False,
                verbose=False,
                agnostic_nms=True,
            )

        assert model.predict.call_args_list == [
            create_call(input_frames[1][FrameKeys.data]),
            create_call(input_frames[2][FrameKeys.data]),
        ]

    def assert_detected_frame_factory_called(
        self, factory: Mock, input_frames: list[Frame], detections: list[Detection]
    ) -> None:
        assert factory.create.call_args_list == [
            call(input_frames[0], detections=[]),
            call(input_frames[1], detections=detections[0]),
            call(input_frames[2], detections=detections[1]),
            call(input_frames[3], detections=[]),
        ]

    def assert_detection_converter_called(
        self,
        detection_converter: Mock,
        model_predictions: list[Results],
        normalized: bool,
        class_mapping: dict[int, str],
    ) -> None:
        assert detection_converter.convert.call_args_list == [
            call(
                model_predictions[0].boxes,
                normalized=normalized,
                classification_mapping=class_mapping,
            ),
            call(
                model_predictions[1].boxes,
                normalized=normalized,
                classification_mapping=class_mapping,
            ),
        ]

    def create_model(self, model_predictions: list[Results]) -> Mock:
        model = Mock()
        model.predict.side_effect = [[prediction] for prediction in model_predictions]
        return model

    def create_get_current_config(self, config: Config) -> Mock:
        get_current_config = Mock()
        get_current_config.get.return_value = config
        return get_current_config

    def create_detection_converter(self, detections: list[Detection]) -> Mock:
        detection_converter = Mock()
        detection_converter.convert.side_effect = detections
        return detection_converter

    def create_detected_frame_factory(
        self, detected_frames: list[DetectedFrame]
    ) -> Mock:
        factory = Mock()
        factory.create.side_effect = detected_frames
        return factory

    def creat_input_frames(self) -> list[Frame]:
        first = self.create_mock_detection(False, 0)
        second = self.create_mock_detection(True, 1)
        third = self.create_mock_detection(True, 2)
        last = self.create_mock_detection(False, 2)
        return [first, second, third, last]

    def create_mock_detection(
        self, has_data: bool, frame_no: int, source: str = SOURCE
    ) -> Frame:
        data = None
        if has_data:
            data = Mock()

        return Frame(data=data, frame=frame_no, source=source, occurrence=Mock())
