from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self

from OTVision import dataformat, version
from OTVision.domain.detection import DetectedFrame, Detection


@dataclass
class OtdetBuilderConfig:
    conf: float
    iou: float
    source: str
    video_width: int
    video_height: int
    expected_duration: timedelta | None
    actual_duration: timedelta
    recorded_fps: float
    recorded_start_date: datetime
    actual_fps: float
    actual_frames: int
    detection_img_size: int
    normalized: bool
    detection_model: str | Path
    half_precision: bool
    chunksize: int
    classifications: dict[int, str]
    detect_start: int | None
    detect_end: int | None


class OtdetBuilderError(Exception):
    pass


class OtdetBuilder:
    @property
    def config(self) -> OtdetBuilderConfig:
        if self._config is None:
            raise OtdetBuilderError("Otdet builder config is not set")
        return self._config

    def __init__(self) -> None:
        self._config: OtdetBuilderConfig | None = None

    def add_config(self, config: OtdetBuilderConfig) -> Self:
        self._config = config
        return self

    def reset(self) -> Self:
        self._config = None
        return self

    def build(self, detections: list[DetectedFrame]) -> dict:
        number_of_frames = len(detections)
        result = {
            dataformat.METADATA: self._build_metadata(number_of_frames),
            dataformat.DATA: self._build_data(detections),
        }
        self.reset()
        return result

    def _build_metadata(self, number_of_frames: int) -> dict:
        return {
            dataformat.OTDET_VERSION: version.otdet_version(),
            dataformat.VIDEO: self._build_video_config(number_of_frames),
            dataformat.DETECTION: self._build_detection_config(),
        }

    def _build_data(self, frames: list[DetectedFrame]) -> dict:
        data = {}
        for frame in frames:
            converted_detections = [
                self.__convert_detection(detection) for detection in frame.detections
            ]
            data[str(frame.frame_number)] = {
                dataformat.DETECTIONS: converted_detections,
                dataformat.OCCURRENCE: frame.occurrence.timestamp(),
            }
        return data

    def __convert_detection(self, detection: Detection) -> dict:
        return {
            dataformat.CLASS: detection.label,
            dataformat.CONFIDENCE: detection.conf,
            dataformat.X: detection.x,
            dataformat.Y: detection.y,
            dataformat.W: detection.w,
            dataformat.H: detection.h,
        }

    def _build_video_config(self, number_of_frames: int) -> dict:
        source = Path(self.config.source)
        video_config = {
            dataformat.FILENAME: str(source.stem),
            dataformat.FILETYPE: str(source.suffix),
            dataformat.WIDTH: self.config.video_width,
            dataformat.HEIGHT: self.config.video_height,
            dataformat.RECORDED_FPS: self.config.recorded_fps,
            dataformat.ACTUAL_FPS: self.config.actual_fps,
            dataformat.NUMBER_OF_FRAMES: number_of_frames,
            dataformat.RECORDED_START_DATE: self.config.recorded_start_date.timestamp(),
            dataformat.LENGTH: str(self.config.actual_duration),
        }
        if self.config.expected_duration is not None:
            video_config[dataformat.EXPECTED_DURATION] = int(
                self.config.expected_duration.total_seconds()
            )
        return video_config

    def _build_detection_config(self) -> dict:
        return {
            dataformat.OTVISION_VERSION: version.otvision_version(),
            dataformat.MODEL: {
                dataformat.NAME: "YOLOv8",
                dataformat.WEIGHTS: str(self.config.detection_model),
                dataformat.IOU_THRESHOLD: self.config.iou,
                dataformat.IMAGE_SIZE: self.config.detection_img_size,
                dataformat.MAX_CONFIDENCE: self.config.conf,
                dataformat.HALF_PRECISION: self.config.half_precision,
                dataformat.CLASSES: self.config.classifications,
            },
            dataformat.CHUNKSIZE: self.config.chunksize,
            dataformat.NORMALIZED_BBOX: self.config.normalized,
            dataformat.DETECT_START: self.config.detect_start,
            dataformat.DETECT_END: self.config.detect_end,
        }
