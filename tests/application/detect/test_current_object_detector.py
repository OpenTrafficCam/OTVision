from unittest.mock import Mock

from OTVision.application.detect.current_object_detector import CurrentObjectDetector
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.config import Config
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

    def test_filter(self) -> None:
        expected_generator = Mock()
        given_input_generator = Mock()
        given_object_detector = create_object_detector(expected_generator)
        given_get_current_config = create_get_current_config()
        given_factory = create_factory(given_object_detector)

        target = CurrentObjectDetector(given_get_current_config, given_factory)
        actual = target.filter(given_input_generator)
        assert actual == expected_generator


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
