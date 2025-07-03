from functools import cached_property

from OTVision.application.video.generate_video import GenerateVideo
from OTVision.detect.file_based_detect_builder import FileBasedDetectBuilder


class GenerateVideoBuilder(FileBasedDetectBuilder):
    @cached_property
    def generate_video(self) -> GenerateVideo:
        return GenerateVideo(
            input_source=self.input_source, video_writer=self.video_file_writer
        )

    def build_generate_video(self) -> GenerateVideo:
        self.register_observers()
        return self.generate_video

    def register_observers(self) -> None:
        self.input_source.subject_new_video_start.register(
            self.video_file_writer.notify_on_new_video_start
        )
        self.input_source.subject_flush.register(
            self.video_file_writer.notify_on_flush_event
        )
