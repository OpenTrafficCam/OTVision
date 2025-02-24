from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Self

from OTVision import dataformat, version
from OTVision.track.preprocess import Detection


@dataclass
class OtdetBuilderConfig:
    conf: float
    iou: float
    video: Path
    video_width: int
    video_height: int
    expected_duration: timedelta | None
    recorded_fps: float
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

    def build(self, detections: list[list[Detection]]) -> dict:
        result = {
            dataformat.METADATA: self._build_metadata(),
            dataformat.DATA: self._build_data(detections),
        }
        self.reset()
        return result

    def _build_metadata(self) -> dict:
        return {
            dataformat.OTDET_VERSION: version.otdet_version(),
            dataformat.VIDEO: self._build_video_config(),
            dataformat.DETECTION: self._build_detection_config(),
        }

    def _build_data(self, frames: list[list[Detection]]) -> dict:
        data = {}
        for frame, detections in enumerate(frames, start=1):
            converted_detections = [detection.to_otdet() for detection in detections]
            data[str(frame)] = {dataformat.DETECTIONS: converted_detections}
        return data

    def _build_video_config(self) -> dict:
        video_config = {
            dataformat.FILENAME: str(self.config.video.stem),
            dataformat.FILETYPE: str(self.config.video.suffix),
            dataformat.WIDTH: self.config.video_width,
            dataformat.HEIGHT: self.config.video_height,
            dataformat.RECORDED_FPS: self.config.recorded_fps,
            dataformat.ACTUAL_FPS: self.config.actual_fps,
            dataformat.NUMBER_OF_FRAMES: self.config.actual_frames,
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
