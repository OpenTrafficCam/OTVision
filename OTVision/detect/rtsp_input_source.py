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
        self._start_value = start_value
        self.__counter = start_value

    def increment(self) -> None:
        self.__counter += 1

    def get(self) -> int:
        return self.__counter

    def reset(self) -> None:
        self.__counter = self._start_value


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

    @property
    def flush_buffer_size(self) -> int:
        return self.stream_config.flush_buffer_size

    @property
    def fps(self) -> float:
        return self.config.convert.output_fps

    def __init__(
        self,
        subject: Subject[FlushEvent],
        datetime_provider: DatetimeProvider,
        frame_counter: Counter,
        get_current_config: GetCurrentConfig,
    ) -> None:
        super().__init__(subject)
        self._datetime_provider = datetime_provider
        self._stop_capture = False
        self._frame_counter = frame_counter
        self._get_current_config = get_current_config
        self._current_stream: str | None = None
        self._current_video_capture: VideoCapture | None = None
        self._stream_start_time: datetime = self._datetime_provider.provide()
        self._current_video_start_time = self._stream_start_time
        self._outdated = True

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
        self._current_video_capture = self._init_video_capture(self._current_stream)
        return self._current_video_capture

    def produce(self) -> Generator[Frame, None, None]:
        self._stream_start_time = self._datetime_provider.provide()
        self._current_video_start_time = self._stream_start_time
        while not self.should_stop():
            if (frame := self._read_next_frame()) is not None:
                self._frame_counter.increment()
                occurrence = self._datetime_provider.provide()

                if self._outdated:
                    self._current_video_start_time = occurrence
                    self._outdated = False

                yield Frame(
                    data=convert_frame_to_rgb(frame),  # YOLO expects RGB
                    frame=self.current_frame_number,
                    source=self.rtsp_url,
                    output=self.create_output(),
                    occurrence=occurrence,
                )
                if self.flush_condition_met():
                    self._notify()
                    self._outdated = True
                    self._frame_counter.reset()

        self._notify()

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
        return self.current_frame_number % self.flush_buffer_size == 0

    def _notify(self) -> None:
        frame_width = int(self._video_capture.get(CAP_PROP_FRAME_WIDTH))
        frame_height = int(self._video_capture.get(CAP_PROP_FRAME_HEIGHT))
        frames = (
            self.flush_buffer_size
            if self.current_frame_number % self.flush_buffer_size == 0
            else self.current_frame_number % self.flush_buffer_size
        )
        duration = timedelta(seconds=round(frames / self.fps))
        output = self.create_output()
        self._subject.notify(
            FlushEvent.create(
                source=self.rtsp_url,
                output=output,
                duration=duration,
                source_width=frame_width,
                source_height=frame_height,
                source_fps=self.fps,
                start_time=self._current_video_start_time,
            )
        )

    def create_output(self) -> str:
        output_filename = (
            f"{self.stream_config.name}_FR{round(self.fps)}"
            f"_{self._current_video_start_time.strftime(DATETIME_FORMAT)}.mp4"
        )
        return str(self.stream_config.save_dir / output_filename)


def convert_frame_to_rgb(frame: ndarray) -> ndarray:
    return cvtColor(frame, COLOR_BGR2RGB)
