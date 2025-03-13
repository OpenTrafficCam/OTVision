from argparse import ArgumentParser
from functools import cached_property

from OTVision.abstraction.observer import Subject
from OTVision.application.buffer import Buffer
from OTVision.application.configure_logger import ConfigureLogger
from OTVision.application.detect.current_object_detector import CurrentObjectDetector
from OTVision.application.detect.current_object_detector_metadata import (
    CurrentObjectDetectorMetadata,
)
from OTVision.application.detect.detected_frame_factory import DetectedFrameFactory
from OTVision.application.detect.detected_frame_producer import (
    SimpleDetectedFrameProducer,
)
from OTVision.application.detect.detection_file_save_path_provider import (
    DetectionFileSavePathProvider,
)
from OTVision.application.detect.factory import ObjectDetectorCachedFactory
from OTVision.application.detect.get_detect_cli_args import GetDetectCliArgs
from OTVision.application.detect.update_detect_config_with_cli_args import (
    UpdateDetectConfigWithCliArgs,
)
from OTVision.application.frame_count_provider import FrameCountProvider
from OTVision.application.get_config import GetConfig
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.application.update_current_config import UpdateCurrentConfig
from OTVision.config import Config, ConfigParser
from OTVision.detect.cli import ArgparseDetectCliParser
from OTVision.detect.detect import OTVisionVideoDetect
from OTVision.detect.detected_frame_buffer import (
    DetectedFrameBuffer,
    DetectedFrameBufferEvent,
    FlushEvent,
)
from OTVision.detect.otdet import OtdetBuilder
from OTVision.detect.otdet_file_writer import OtdetFileWriter
from OTVision.detect.plugin_av.rotate_frame import AvVideoFrameRotator
from OTVision.detect.pyav_frame_count_provider import PyAVFrameCountProvider
from OTVision.detect.timestamper import TimestamperFactory
from OTVision.detect.video_input_source import VideoSource
from OTVision.detect.yolo import YoloDetectionConverter, YoloFactory
from OTVision.domain.cli import DetectCliParser
from OTVision.domain.current_config import CurrentConfig
from OTVision.domain.detect_producer_consumer import DetectedFrameProducer
from OTVision.domain.detection import DetectedFrame
from OTVision.domain.input_source_detect import InputSourceDetect
from OTVision.domain.object_detection import ObjectDetectorFactory


class DetectBuilder:
    @cached_property
    def get_config(self) -> GetConfig:
        return GetConfig(self.config_parser)

    @cached_property
    def config_parser(self) -> ConfigParser:
        return ConfigParser()

    @cached_property
    def get_detect_cli_args(self) -> GetDetectCliArgs:
        return GetDetectCliArgs(self.detect_cli_parser)

    @cached_property
    def detect_cli_parser(self) -> DetectCliParser:
        return ArgparseDetectCliParser(
            parser=ArgumentParser("Detect objects in videos or images"), argv=self.argv
        )

    @cached_property
    def update_detect_config_with_ci_args(self) -> UpdateDetectConfigWithCliArgs:
        return UpdateDetectConfigWithCliArgs(self.get_detect_cli_args)

    @cached_property
    def configure_logger(self) -> ConfigureLogger:
        return ConfigureLogger()

    @cached_property
    def otdet_builder(self) -> OtdetBuilder:
        return OtdetBuilder()

    @cached_property
    def object_detector_factory(self) -> ObjectDetectorFactory:
        return ObjectDetectorCachedFactory(
            YoloFactory(
                get_current_config=self.get_current_config,
                detection_converter=self.detection_converter,
                detected_frame_factory=self.frame_converter,
            )
        )

    @cached_property
    def detection_converter(self) -> YoloDetectionConverter:
        return YoloDetectionConverter()

    @cached_property
    def frame_converter(self) -> DetectedFrameFactory:
        return DetectedFrameFactory()

    @cached_property
    def current_config(self) -> CurrentConfig:
        return CurrentConfig(Config())

    @cached_property
    def get_current_config(self) -> GetCurrentConfig:
        return GetCurrentConfig(self.current_config)

    @cached_property
    def update_current_config(self) -> UpdateCurrentConfig:
        return UpdateCurrentConfig(self.current_config)

    def __init__(self, argv: list[str] | None = None) -> None:
        self.argv = argv

    @cached_property
    def input_source(self) -> InputSourceDetect:
        return VideoSource(
            subject=Subject[FlushEvent](),
            get_current_config=self.get_current_config,
            frame_rotator=self.frame_rotator,
            timestamper_factory=self.timestamper_factory,
            save_path_provider=self.detection_file_save_path_provider,
        )

    @cached_property
    def frame_rotator(self) -> AvVideoFrameRotator:
        return AvVideoFrameRotator()

    @cached_property
    def timestamper_factory(self) -> TimestamperFactory:
        return TimestamperFactory(self.frame_count_provider, self.get_current_config)

    @cached_property
    def detection_file_save_path_provider(self) -> DetectionFileSavePathProvider:
        return DetectionFileSavePathProvider(self.get_current_config)

    @cached_property
    def frame_count_provider(self) -> FrameCountProvider:
        return PyAVFrameCountProvider()

    @cached_property
    def otdet_file_writer(self) -> OtdetFileWriter:
        return OtdetFileWriter(
            builder=self.otdet_builder,
            get_current_config=self.get_current_config,
            current_object_detector_metadata=self.current_object_detector_metadata,
            save_path_provider=self.detection_file_save_path_provider,
        )

    @cached_property
    def current_object_detector_metadata(self) -> CurrentObjectDetectorMetadata:
        return CurrentObjectDetectorMetadata(self.current_object_detector)

    @cached_property
    def current_object_detector(self) -> CurrentObjectDetector:
        return CurrentObjectDetector(
            get_current_config=self.get_current_config,
            factory=self.object_detector_factory,
        )

    @cached_property
    def detected_frame_buffer(
        self,
    ) -> Buffer[DetectedFrame, DetectedFrameBufferEvent, FlushEvent]:
        return DetectedFrameBuffer(subject=Subject[DetectedFrameBufferEvent]())

    @cached_property
    def detected_frame_producer(self) -> DetectedFrameProducer:
        return SimpleDetectedFrameProducer(
            input_source=self.input_source,
            detection_filter=self.current_object_detector,
            detected_frame_buffer=self.detected_frame_buffer,
        )

    def build(self) -> OTVisionVideoDetect:
        self.register_observers()
        return OTVisionVideoDetect(self.detected_frame_producer)

    def register_observers(self) -> None:
        self.input_source.register(self.detected_frame_buffer.on_flush)
        self.detected_frame_buffer.register(self.otdet_file_writer.write)
