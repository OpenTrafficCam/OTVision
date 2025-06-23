from typing import Generator

from OTVision.abstraction.pipes_and_filter import Filter
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.domain.frame import DetectedFrame, Frame
from OTVision.domain.input_source_detect import InputSourceDetect


class DetectedFrameProducerFactory:
    def __init__(
        self,
        input_source: InputSourceDetect,
        video_writer_filter: Filter[Frame, Frame],
        detection_filter: Filter[Frame, DetectedFrame],
        detected_frame_buffer: Filter[DetectedFrame, DetectedFrame],
        get_current_config: GetCurrentConfig,
    ) -> None:
        self._input_source = input_source
        self._video_writer_filter = video_writer_filter
        self._detection_filter = detection_filter
        self._detected_frame_buffer = detected_frame_buffer
        self._get_current_config = get_current_config

    def create(self) -> Generator[DetectedFrame, None, None]:
        if self._get_current_config.get().detect.write_video:
            return self.__create_with_video_writer()
        return self.__create_without_video_writer()

    def __create_without_video_writer(
        self,
    ) -> Generator[DetectedFrame, None, None]:
        return self._detected_frame_buffer.filter(
            self._detection_filter.filter(self._input_source.produce())
        )

    def __create_with_video_writer(
        self,
    ) -> Generator[DetectedFrame, None, None]:
        return self._detected_frame_buffer.filter(
            self._detection_filter.filter(
                self._video_writer_filter.filter(self._input_source.produce())
            )
        )
