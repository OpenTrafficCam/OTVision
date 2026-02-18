from typing import AsyncIterator
from unittest.mock import Mock

import pytest

from OTVision.application.config import Config
from OTVision.application.detect.current_object_detector import CurrentObjectDetector
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.domain.frame import DetectedFrame, Frame
from OTVision.domain.object_detection import ObjectDetector, ObjectDetectorFactory

CONFIG = Config()


class TestCurrentObjectDetector:
    def test_get(self) -> None:
        expected = create_object_detector(Mock())
        given_get_current_config = create_get_current_config()
        given_factory = create_factory(expected)

        target = CurrentObjectDetector(given_get_current_config, given_factory)
        actual = target.get()
        assert actual == expected

    @pytest.mark.asyncio
    async def test_filter(self) -> None:
        given_detected_frames = [Mock(spec=DetectedFrame), Mock(spec=DetectedFrame)]

        async def mock_detect_generator(
            frames: AsyncIterator[Frame],
        ) -> AsyncIterator[DetectedFrame]:
            for frame in given_detected_frames:
                yield frame

        async def input_generator() -> AsyncIterator[Frame]:
            yield Mock(spec=Frame)
            yield Mock(spec=Frame)

        given_object_detector = Mock(spec=ObjectDetector)
        given_object_detector.detect = mock_detect_generator
        given_get_current_config = create_get_current_config()
        given_factory = create_factory(given_object_detector)

        target = CurrentObjectDetector(given_get_current_config, given_factory)
        actual_frames = []
        async for frame in target.filter(input_generator()):
            actual_frames.append(frame)

        assert actual_frames == given_detected_frames


def create_get_current_config() -> Mock:
    mock = Mock(spec=GetCurrentConfig)
    mock.get.return_value = CONFIG
    return mock


def create_factory(object_detector: Mock) -> Mock:
    mock = Mock(spec=ObjectDetectorFactory)
    mock.create.return_value = object_detector
    return mock


def create_object_detector(generator: Mock) -> Mock:
    mock = Mock(spec=ObjectDetector)
    mock.detect.return_value = generator
    return mock
