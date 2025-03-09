from typing import Generator

from OTVision.abstraction.pipes_and_filter import Filter
from OTVision.domain.detect_producer_consumer import DetectedFrameProducer
from OTVision.domain.detection import DetectedFrame
from OTVision.domain.frame import Frame
from OTVision.domain.input_source_detect import InputSourceDetect


class SimpleDetectedFrameProducer(DetectedFrameProducer):

    def __init__(
        self,
        input_source: InputSourceDetect,
        detection_filter: Filter[Frame, DetectedFrame],
        detected_frame_buffer: Filter[DetectedFrame, DetectedFrame],
    ) -> None:
        self._input_source = input_source
        self._detection_filter = detection_filter
        self._detected_frame_buffer = detected_frame_buffer

    def produce(self) -> Generator[DetectedFrame, None, None]:
        return self._detected_frame_buffer.filter(
            self._detection_filter.filter(self._input_source.produce())
        )
