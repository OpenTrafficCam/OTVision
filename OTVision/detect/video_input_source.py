import logging
from datetime import datetime
from pathlib import Path
from typing import Generator

import av
from tqdm import tqdm

from OTVision.abstraction.observer import Subject
from OTVision.application.configure_logger import logger
from OTVision.application.detect.detection_file_save_path_provider import (
    DetectionFileSavePathProvider,
)
from OTVision.application.detect.timestamper import Timestamper
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.config import DATETIME_FORMAT, Config
from OTVision.detect.detected_frame_buffer import FlushEvent
from OTVision.detect.plugin_av.rotate_frame import AvVideoFrameRotator
from OTVision.detect.timestamper import TimestamperFactory, parse_start_time_from
from OTVision.domain.frame import FrameKeys
from OTVision.domain.input_source_detect import Frame, InputSourceDetect
from OTVision.helpers.files import InproperFormattedFilename, get_files
from OTVision.helpers.log import LOGGER_NAME
from OTVision.helpers.video import (
    convert_seconds_to_frames,
    get_duration,
    get_fps,
    get_video_dimensions,
)

log = logging.getLogger(LOGGER_NAME)


class VideoSource(InputSourceDetect):
    """A video source that manages video file processing and detection operations.

    This class handles video file processing, including frame generation, detection
    configuration, and observer notifications. It supports video rotation, timestamping,
    and selective frame processing based on configuration parameters.

    Args:
        get_current_config (GetCurrentConfig): Use case to retrieve current
            configuration.
        frame_rotator (AvVideoFrameRotator): Use to rotate video frames.
        timestamper_factory (Timestamper): Factory for creating timestamp generators.
        save_path_provider (DetectionFileSavePathProvider): Provider for detection
            output paths.
    """

    @property
    def _current_config(self) -> Config:
        return self._get_current_config.get()

    @property
    def _start_time(self) -> datetime | None:
        return self._get_current_config.get().detect.start_time

    def __init__(
        self,
        subject: Subject[FlushEvent],
        get_current_config: GetCurrentConfig,
        frame_rotator: AvVideoFrameRotator,
        timestamper_factory: TimestamperFactory,
        save_path_provider: DetectionFileSavePathProvider,
    ) -> None:
        super().__init__(subject)
        self._frame_rotator = frame_rotator
        self._get_current_config = get_current_config
        self._timestamper_factory = timestamper_factory
        self._save_path_provider = save_path_provider
        self.__should_flush = False

    def produce(self) -> Generator[Frame, None, None]:
        """Generate frames from video files that meet detection requirements.

        Yields frames from valid video files while managing rotation, timestamping,
        and detection configurations. Notifies observers about processing events.

        Yields:
            Frame: Processed video frames ready for detection.
        """

        video_files = self.__collect_files_to_detect()

        start_msg = f"Start detection of {len(video_files)} video files"
        log.info(start_msg)
        print(start_msg)

        for video_file in tqdm(video_files, desc="Detected video files", unit=" files"):
            detections_file = self._save_path_provider.provide(str(video_file))

            if not self.__detection_requirements_are_met(video_file, detections_file):
                continue

            # log.info(f"Detect {video_file}")
            timestamper = self._timestamper_factory.create_video_timestamper(
                video_file=video_file,
                expected_duration=self._current_config.detect.expected_duration,
            )
            video_fps = get_fps(video_file)
            detect_start = self.__get_detect_start_in_frames(video_fps)
            detect_end = self.__get_detect_end_in_frames(video_fps)
            counter = 0
            try:
                with av.open(str(video_file.absolute())) as container:
                    container.streams.video[0].thread_type = "AUTO"
                    side_data = container.streams.video[0].side_data
                    for frame_number, frame in enumerate(
                        container.decode(video=0), start=1
                    ):
                        if detect_start <= frame_number and (
                            detect_end is None or frame_number < detect_end
                        ):
                            rotated_image = self._frame_rotator.rotate(frame, side_data)
                            yield timestamper.stamp(
                                {
                                    FrameKeys.data: rotated_image,
                                    FrameKeys.frame: frame_number,
                                    FrameKeys.source: str(video_file),
                                }
                            )
                        else:
                            yield timestamper.stamp(
                                {
                                    FrameKeys.data: None,
                                    FrameKeys.frame: frame_number,
                                    FrameKeys.source: str(video_file),
                                }
                            )
                        counter += 1
                self.notify_observers(video_file, video_fps)
            except Exception as e:
                logger().error(f"Error processing {video_file}", exc_info=e)

    def __collect_files_to_detect(self) -> list[Path]:
        filetypes = self._current_config.filetypes.video_filetypes.to_list()
        video_files = get_files(
            paths=self._current_config.detect.paths, filetypes=filetypes
        )
        if not video_files:
            log.warning(f"No videos of type '{filetypes}' found to detect!")
        return video_files

    def __detection_requirements_are_met(
        self, video_file: Path, detections_file: Path
    ) -> bool:
        return self.__video_file_has_valid_format(
            video_file
        ) and self.__overwrite_existing_detection_file(detections_file)

    def __video_file_has_valid_format(self, video_file: Path) -> bool:
        try:
            parse_start_time_from(video_file, start_time=self._start_time)
            return True
        except InproperFormattedFilename:
            log.warning(
                f"Video file name of '{video_file}' must include date "
                f"and time in format: {DATETIME_FORMAT}"
            )
            return False

    def __overwrite_existing_detection_file(self, detections_file: Path) -> bool:
        if not self._current_config.detect.overwrite and detections_file.is_file():
            log.warning(
                f"{detections_file} already exists. To overwrite, set overwrite "
                "to True"
            )
            return False
        return True

    def notify_observers(self, current_video_file: Path, video_fps: float) -> None:
        if expected_duration := self._current_config.detect.expected_duration:
            duration = expected_duration
        else:
            duration = get_duration(current_video_file)

        width, height = get_video_dimensions(current_video_file)
        start_time = parse_start_time_from(
            current_video_file, start_time=self._start_time
        )

        self._subject.notify(
            FlushEvent.create(
                source=str(current_video_file),
                duration=duration,
                source_height=height,
                source_width=width,
                source_fps=video_fps,
                start_time=start_time,
            )
        )

    def __get_detect_start_in_frames(self, video_fps: float) -> int:
        detect_start = convert_seconds_to_frames(
            self._current_config.detect.detect_start, video_fps
        )
        if detect_start is not None:
            return detect_start
        return 0

    def __get_detect_end_in_frames(self, video_fps: float) -> int | None:
        return convert_seconds_to_frames(
            self._current_config.detect.detect_end, video_fps
        )

    def __add_occurrence(self, timestamper: Timestamper, frame: dict) -> Frame:
        updated = timestamper.stamp(frame)
        return Frame(
            data=updated["data"],
            frame=updated["frame"],
            source=updated["source"],
            occurrence=updated["occurrence"],
        )
