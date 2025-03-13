import logging

from OTVision.application.detect.current_object_detector_metadata import (
    CurrentObjectDetectorMetadata,
)
from OTVision.application.detect.detection_file_save_path_provider import (
    DetectionFileSavePathProvider,
)
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.detect.detected_frame_buffer import DetectedFrameBufferEvent
from OTVision.detect.otdet import OtdetBuilder, OtdetBuilderConfig
from OTVision.helpers.files import write_json
from OTVision.helpers.log import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)


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
        save_path_provider (DetectionFileSavePathProvider): determines the save path for
            the otdet file to be written.

    """

    def __init__(
        self,
        builder: OtdetBuilder,
        get_current_config: GetCurrentConfig,
        current_object_detector_metadata: CurrentObjectDetectorMetadata,
        save_path_provider: DetectionFileSavePathProvider,
    ):
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
        if (expected_duration := detect_config.expected_duration) is not None:
            actual_fps = actual_frames / expected_duration.total_seconds()
        else:
            actual_fps = actual_frames / source_metadata.duration.total_seconds()

        class_mapping = self._current_object_detector_metadata.get().classifications
        otdet = self._builder.add_config(
            OtdetBuilderConfig(
                conf=detect_config.confidence,
                iou=detect_config.iou,
                source=source_metadata.source,
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
        ).build(event.frames)

        detections_file = self._save_path_provider.provide(source_metadata.source)
        write_json(
            otdet,
            file=detections_file,
            filetype=config.filetypes.detect,
            overwrite=detect_config.overwrite,
        )

        log.info(f"Successfully detected and wrote {detections_file}")

        finished_msg = "Finished detection"
        log.info(finished_msg)
