from functools import cached_property

from OTVision.abstraction.observer import Subject
from OTVision.application.config_parser import ConfigParser
from OTVision.detect.builder import DetectBuilder
from OTVision.detect.detected_frame_buffer import FlushEvent
from OTVision.detect.video_input_source import VideoSource
from OTVision.domain.input_source_detect import InputSourceDetect
from OTVision.plugin.config_parser.file_source import FileSourceConfigParser
from OTVision.plugin.yaml_serialization import YamlDeserializer


class FileBasedDetectBuilder(DetectBuilder):
    @cached_property
    def config_parser(self) -> ConfigParser:
        return FileSourceConfigParser(self.yaml_deserializer)

    @cached_property
    def yaml_deserializer(self) -> YamlDeserializer:
        return YamlDeserializer()

    @cached_property
    def input_source(self) -> InputSourceDetect:
        return VideoSource(
            subject=Subject[FlushEvent](),
            get_current_config=self.get_current_config,
            frame_rotator=self.frame_rotator,
            timestamper_factory=self.timestamper_factory,
            save_path_provider=self.detection_file_save_path_provider,
        )
