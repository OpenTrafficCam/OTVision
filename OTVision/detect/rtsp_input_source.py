from datetime import datetime, timedelta
from time import sleep
from typing import Generator

from cv2 import (
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    COLOR_BGR2RGB,
    VideoCapture,
    cvtColor,
)
from numpy import ndarray

from OTVision.abstraction.observer import Subject
from OTVision.application.config import (
    DATETIME_FORMAT,
    Config,
    DetectConfig,
    StreamConfig,
)
from OTVision.application.configure_logger import logger
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.detect.detected_frame_buffer import FlushEvent
from OTVision.domain.frame import Frame
from OTVision.domain.input_source_detect import InputSourceDetect
from OTVision.domain.time import DatetimeProvider

RTSP_URL = "rtsp://127.0.0.1:8554/test"
RETRY_SECONDS = 1


class Counter:
    def __init__(self, start_value: int = 0) -> None:
        self.__counter = start_value

    def increment(self) -> None:
        self.__counter += 1

    def get(self) -> int:
        return self.__counter


class RtspInputSource(InputSourceDetect):

    @property
    def current_frame_number(self) -> int:
        return self._frame_counter.get()

    @property
    def config(self) -> Config:
        return self._get_current_config.get()

    @property
    def detect_config(self) -> DetectConfig:
        return self.config.detect

    @property
    def stream_config(self) -> StreamConfig:
        if stream_config := self.config.stream:
            return stream_config
        raise ValueError("Stream config not found in config")

    @property
    def rtsp_url(self) -> str:
        return self.stream_config.source

    def __init__(
        self,
        subject: Subject[FlushEvent],
        datetime_provider: DatetimeProvider,
        frame_counter: Counter,
        get_current_config: GetCurrentConfig,
        flush_buffer_size: int,
    ) -> None:
        super().__init__(subject)
        self._datetime_provider = datetime_provider
        self._stop_capture = False
        self._frame_counter = frame_counter
        self._flush_buffer_size = flush_buffer_size
        self._get_current_config = get_current_config
        self._current_stream: str | None = None
        self._current_video_capture: VideoCapture | None = None

    @property
    def _video_capture(self) -> VideoCapture:
        # Property is moved below __init__ otherwise mypy is somehow unable to determine
        # types of self._current_stream and self._current_video_capture
        new_source = self.stream_config.source
        if (
            self._current_stream is not None
            and self._current_stream == new_source
            and self._current_video_capture
        ):
            # current source has not changed
            return self._current_video_capture

        # Stream changed or has not been initialized
        if self._current_video_capture is not None:
            # If the stream changed and there's an existing capture, release it
            self._current_video_capture.release()
        self._current_stream = new_source
        self._current_video_capture = VideoCapture(new_source)

        if self._current_video_capture is None:
            raise ValueError("Video capture not initialized")

        self._current_video_capture = self._init_video_capture(self._current_stream)
        return self._current_video_capture

    def produce(self) -> Generator[Frame, None, None]:
        start_time = self._datetime_provider.provide()
        while not self.should_stop():
            if (frame := self._read_next_frame()) is not None:
                self._frame_counter.increment()

                current_frame_number = self.current_frame_number
                yield Frame(
                    data=convert_frame_to_rgb(frame),  # YOLO expects RGB
                    frame=current_frame_number,
                    source=self.rtsp_url,
                    occurrence=self._datetime_provider.provide(),
                )
                if self.flush_condition_met():
                    self._notify(start_time, current_frame_number)
        self._notify(start_time, self.current_frame_number)

    def _init_video_capture(self, source: str) -> VideoCapture:
        cap = VideoCapture(source)
        while not self.should_stop() and not cap.isOpened():
            logger().warning(
                f"Couldn't open the RTSP stream: {source}. "
                f"Trying again in {RETRY_SECONDS}s..."
            )
            sleep(RETRY_SECONDS)
            cap.release()
            cap = VideoCapture(source)
        return cap

    def _read_next_frame(self) -> ndarray | None:
        successful, frame = self._video_capture.read()
        if successful:
            return frame
        logger().debug("Failed to grab frame")
        return None

    def should_stop(self) -> bool:
        return self._stop_capture

    def stop(self) -> None:
        self._stop_capture = True

    def start(self) -> None:
        self._stop_capture = False

    def flush_condition_met(self) -> bool:
        return self.current_frame_number % self._flush_buffer_size == 0

    def _notify(self, start_time: datetime, current_frame_number: int) -> None:
        frame_width = int(self._video_capture.get(CAP_PROP_FRAME_WIDTH))
        frame_height = int(self._video_capture.get(CAP_PROP_FRAME_HEIGHT))
        fps = self.config.convert.output_fps
        _start_time = calculate_start_time(
            start_time, current_frame_number, fps, self._flush_buffer_size
        )
        frames = (
            self._flush_buffer_size
            if current_frame_number % self._flush_buffer_size == 0
            else self.current_frame_number % self._flush_buffer_size
        )
        duration = timedelta(seconds=round(frames / fps))
        output_filename = (
            f"{self.stream_config.name}_FR{round(fps)}"
            f"_{_start_time.strftime(DATETIME_FORMAT)}.mp4"
        )
        output = str(self.stream_config.save_dir / output_filename)
        self._subject.notify(
            FlushEvent.create(
                source=self.rtsp_url,
                output=output,
                duration=duration,
                source_width=frame_width,
                source_height=frame_height,
                source_fps=fps,
                start_time=_start_time,
            )
        )


def calculate_start_time(
    start: datetime, current_frame_number: int, fps: float, flush_buffer_size: int
) -> datetime:
    offset_in_frames = (
        current_frame_number // flush_buffer_size - 1
    ) * flush_buffer_size
    if offset_in_frames == 0:
        return start
    offset_in_seconds = offset_in_frames / fps
    return start + timedelta(seconds=offset_in_seconds)


def convert_frame_to_rgb(frame: ndarray) -> ndarray:
    return cvtColor(frame, COLOR_BGR2RGB)
