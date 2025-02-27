from datetime import timedelta
from pathlib import Path

from OTVision import dataformat, version
from OTVision.track.model.detection import Detection

# from OTVision.track.legacy.preprocess import Detection


class OtdetBuilder:
    def __init__(
        self,
        conf: float,
        iou: float,
        video: Path,
        video_width: int,
        video_height: int,
        expected_duration: timedelta,
        recorded_fps: float,
        actual_fps: float,
        actual_frames: int,
        detection_img_size: int,
        normalized: bool,
        detection_model: str | Path,
        half_precision: bool,
        chunksize: int,
        classifications: dict[int, str],
    ) -> None:
        self._conf = conf
        self._iou = iou
        self._video = video
        self._video_width = video_width
        self._video_height = video_height
        self._expected_duration = expected_duration
        self._recorded_fps = recorded_fps
        self._actual_fps = actual_fps
        self._actual_frames = actual_frames
        self._detection_img_size = detection_img_size
        self._normalized = normalized
        self._detection_model = detection_model
        self._half_precision = half_precision
        self._chunksize = chunksize
        self._classifications = classifications

    def build(self, detections: list[list[Detection]]) -> dict:
        return {
            dataformat.METADATA: self._build_metadata(),
            dataformat.DATA: self._build_data(detections),
        }

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
        return {
            dataformat.FILENAME: str(self._video.stem),
            dataformat.FILETYPE: str(self._video.suffix),
            dataformat.WIDTH: self._video_width,
            dataformat.HEIGHT: self._video_height,
            dataformat.EXPECTED_DURATION: int(self._expected_duration.total_seconds()),
            dataformat.RECORDED_FPS: self._recorded_fps,
            dataformat.ACTUAL_FPS: self._actual_fps,
            dataformat.NUMBER_OF_FRAMES: self._actual_frames,
        }

    def _build_detection_config(self) -> dict:
        return {
            dataformat.OTVISION_VERSION: version.otvision_version(),
            dataformat.MODEL: {
                dataformat.NAME: "YOLOv8",
                dataformat.WEIGHTS: str(self._detection_model),
                dataformat.IOU_THRESHOLD: self._iou,
                dataformat.IMAGE_SIZE: self._detection_img_size,
                dataformat.MAX_CONFIDENCE: self._conf,
                dataformat.HALF_PRECISION: self._half_precision,
                dataformat.CLASSES: self._classifications,
            },
            dataformat.CHUNKSIZE: self._chunksize,
            dataformat.NORMALIZED_BBOX: self._normalized,
        }
