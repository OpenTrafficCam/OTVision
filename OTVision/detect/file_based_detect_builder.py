from functools import cached_property

from OTVision.abstraction.observer import Subject
from OTVision.application.event.new_video_start import NewVideoStartEvent
from OTVision.detect.builder import DetectBuilder
from OTVision.detect.detected_frame_buffer import FlushEvent
from OTVision.detect.video_input_source import VideoSource


class FileBasedDetectBuilder(DetectBuilder):

    @cached_property
    def input_source(self) -> VideoSource:
        return VideoSource(
            subject_flush=Subject[FlushEvent](),
            subject_new_video_start=Subject[NewVideoStartEvent](),
            get_current_config=self.get_current_config,
            frame_rotator=self.frame_rotator,
            timestamper_factory=self.timestamper_factory,
            save_path_provider=self.detection_file_save_path_provider,
        )

    def register_observers(self) -> None:
        self.input_source.subject_new_video_start.register(
            self.video_file_writer.notify_on_new_video_start
        )
        self.input_source.subject_flush.register(
            self.video_file_writer.notify_on_flush_event
        )
        self.input_source.subject_flush.register(self.detected_frame_buffer.on_flush)
        self.detected_frame_buffer.register(self.otdet_file_writer.write)
