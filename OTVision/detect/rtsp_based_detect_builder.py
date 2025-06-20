from functools import cached_property

from OTVision.abstraction.observer import Subject
from OTVision.application.config import StreamConfig
from OTVision.application.event.new_video_start import NewVideoStartEvent
from OTVision.detect.builder import DetectBuilder
from OTVision.detect.detected_frame_buffer import FlushEvent
from OTVision.detect.rtsp_input_source import Counter, RtspInputSource
from OTVision.domain.time import CurrentDatetimeProvider, DatetimeProvider

FLUSH_BUFFER_SIZE = 18000
FLUSH_BUFFER_SIZE = 1200


class RtspBasedDetectBuilder(DetectBuilder):
    @property
    def stream_config(self) -> StreamConfig:
        config = self.get_current_config.get()
        if config.stream is None:
            raise ValueError(
                "Stream config is not provided. "
                "Running OTVision in streaming mode requires stream config"
            )
        return config.stream

    @cached_property
    def input_source(self) -> RtspInputSource:
        return RtspInputSource(
            subject_flush=Subject[FlushEvent](),
            subject_new_video_start=Subject[NewVideoStartEvent](),
            datetime_provider=self.datetime_provider,
            frame_counter=Counter(),
            get_current_config=self.get_current_config,
        )

    @cached_property
    def datetime_provider(self) -> DatetimeProvider:
        return CurrentDatetimeProvider()

    def register_observers(self) -> None:
        self.input_source.subject_flush.register(self.detected_frame_buffer.on_flush)
        self.detected_frame_buffer.register(self.otdet_file_writer.write)
