from unittest.mock import Mock

from OTVision.detect.detected_frame_producer import SimpleDetectedFrameProducer
from OTVision.detect.detected_frame_producer_factory import DetectedFrameProducerFactory

PRODUCER = Mock()


class TestSimpleDetectedFrameProducer:
    def test_produce(self) -> None:
        given_factory = create_producer_factory()

        target = SimpleDetectedFrameProducer(given_factory)
        actual = target.produce()

        assert actual == PRODUCER


def create_producer_factory() -> Mock:
    mock = Mock(spec=DetectedFrameProducerFactory)
    mock.create.return_value = PRODUCER
    return mock
