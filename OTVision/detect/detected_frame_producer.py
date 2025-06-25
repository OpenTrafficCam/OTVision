from typing import Generator

from OTVision.detect.detected_frame_producer_factory import DetectedFrameProducerFactory
from OTVision.domain.detect_producer_consumer import DetectedFrameProducer
from OTVision.domain.frame import DetectedFrame


class SimpleDetectedFrameProducer(DetectedFrameProducer):

    def __init__(self, producer_factory: DetectedFrameProducerFactory) -> None:
        self._producer_factory = producer_factory

    def produce(self) -> Generator[DetectedFrame, None, None]:
        return self._producer_factory.create()
