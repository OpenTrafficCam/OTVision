import logging
from dataclasses import dataclass
from pathlib import Path

from OTVision.abstraction.observer import Observer, Subject
from OTVision.application.detect.current_object_detector_metadata import (
    CurrentObjectDetectorMetadata,
)
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.application.otvision_save_path_provider import OtvisionSavePathProvider
from OTVision.detect.detected_frame_buffer import DetectedFrameBufferEvent
from OTVision.detect.otdet import OtdetBuilder, OtdetBuilderConfig
from OTVision.helpers.files import write_json
from OTVision.helpers.log import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class OtdetFileWrittenEvent:
    """Event that is emitted when an OTDET file is written."""

    otdet_builder_config: OtdetBuilderConfig
    number_of_frames: int
    save_location: Path


class OtdetFileWriter:
    """Handles writing object detection results to a file in OTDET format.

    This class coordinates the process of building and saving object detection results.
    It combines detection metadata, configuration settings, and frame data to create
    and save OTDET format files.

    Args:
        builder (OtdetBuilder): Responsible for constructing datat to be written.
        get_current_config (GetCurrentConfig): Provides access to current configuration
            settings.
        current_object_detector_metadata (CurrentObjectDetectorMetadata): Provides
            metadata about the current object detector.
        save_path_provider (OtvisionSavePathProvider): determines the save path for
            the otdet file to be written.

    """

    def __init__(
        self,
        subject: Subject[OtdetFileWrittenEvent],
        builder: OtdetBuilder,
        get_current_config: GetCurrentConfig,
        current_object_detector_metadata: CurrentObjectDetectorMetadata,
        save_path_provider: OtvisionSavePathProvider,
    ):
        self._subject = subject
        self._builder = builder
        self._get_current_config = get_current_config
        self._current_object_detector_metadata = current_object_detector_metadata
        self._save_path_provider = save_path_provider

    def write(self, event: DetectedFrameBufferEvent) -> None:
        """Writes detection results to a file in OTDET format.

        Processes the detected frames and associated metadata, builds the OTDET
        structure, and saves it to a file.

        Args:
            event (DetectedFrameBufferEvent): Contains the frames and source metadata
                necessary to build otdet data.

        """

        source_metadata = event.source_metadata
        config = self._get_current_config.get()
        detect_config = config.detect

        actual_frames = len(event.frames)
        if expected_duration := detect_config.expected_duration:
            actual_fps = actual_frames / expected_duration.total_seconds()
        else:
            actual_fps = actual_frames / source_metadata.duration.total_seconds()

        class_mapping = self._current_object_detector_metadata.get().classifications
        builder_config = OtdetBuilderConfig(
            conf=detect_config.confidence,
            iou=detect_config.iou,
            source=source_metadata.output,
            video_width=source_metadata.width,
            video_height=source_metadata.height,
            expected_duration=expected_duration,
            actual_duration=source_metadata.duration,
            recorded_fps=source_metadata.fps,
            recorded_start_date=source_metadata.start_time,
            actual_fps=actual_fps,
            actual_frames=actual_frames,
            detection_img_size=detect_config.img_size,
            normalized=detect_config.normalized,
            detection_model=detect_config.weights,
            half_precision=detect_config.half_precision,
            chunksize=1,
            classifications=class_mapping,
            detect_start=detect_config.detect_start,
            detect_end=detect_config.detect_end,
        )
        otdet = self._builder.add_config(builder_config).build(event.frames)

        detections_file = self._save_path_provider.provide(
            source_metadata.output, config.filetypes.detect
        )
        detections_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            otdet,
            file=detections_file,
            filetype=config.filetypes.detect,
            overwrite=detect_config.overwrite,
        )

        log.info(f"Successfully detected and wrote {detections_file}")

        finished_msg = "Finished detection"
        log.info(finished_msg)
        self.__notify(
            num_frames=actual_frames,
            builder_config=builder_config,
            save_location=detections_file,
        )

    def __notify(
        self, num_frames: int, builder_config: OtdetBuilderConfig, save_location: Path
    ) -> None:
        self._subject.notify(
            OtdetFileWrittenEvent(
                number_of_frames=num_frames,
                otdet_builder_config=builder_config,
                save_location=save_location,
            )
        )

    def register_observer(self, observer: Observer[OtdetFileWrittenEvent]) -> None:
        """Register an observer to receive notifications about otdet file writes.."""
        self._subject.register(observer)
