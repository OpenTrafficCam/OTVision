from functools import cached_property

from OTVision.abstraction.observer import Subject
from OTVision.application.config import StreamConfig
from OTVision.application.event.new_video_start import NewVideoStartEvent
from OTVision.detect.builder import DetectBuilder
from OTVision.detect.detected_frame_buffer import FlushEvent
from OTVision.detect.rtsp_input_source import Counter, RtspInputSource
from OTVision.domain.time import CurrentDatetimeProvider, DatetimeProvider
from OTVision.domain.video_writer import VideoWriter
from OTVision.plugin.ffmpeg_video_writer import (
    FfmpegVideoWriter,
    PixelFormat,
    VideoFormat,
    keep_original_save_location,
)

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

    @cached_property
    def video_file_writer(self) -> VideoWriter:
        return FfmpegVideoWriter(
            save_location_strategy=keep_original_save_location,
            encoding_speed=self.detect_config.encoding_speed,
            input_format=VideoFormat.RAW,
            output_format=VideoFormat.MP4,
            input_pixel_format=PixelFormat.RGB24,
            output_pixel_format=PixelFormat.YUV420P,
            output_video_codec=self.detect_config.video_codec,
            constant_rate_factor=self.detect_config.crf,
        )

    def register_observers(self) -> None:
        if self.detect_config.write_video:
            self.input_source.subject_new_video_start.register(
                self.video_file_writer.notify_on_new_video_start
            )
            self.input_source.subject_flush.register(
                self.video_file_writer.notify_on_flush_event
            )
        self.input_source.subject_flush.register(self.detected_frame_buffer.on_flush)
        self.detected_frame_buffer.register(self.otdet_file_writer.write)
