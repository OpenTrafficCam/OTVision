import socket
from datetime import datetime, timedelta
from time import sleep
from typing import Generator
from urllib.parse import urlparse

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
from OTVision.application.event.new_video_start import NewVideoStartEvent
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.detect.detected_frame_buffer import FlushEvent
from OTVision.domain.frame import Frame
from OTVision.domain.input_source_detect import InputSourceDetect
from OTVision.domain.time import DatetimeProvider

RTSP_URL = "rtsp://127.0.0.1:8554/test"
RETRY_SECONDS = 5
DEFAULT_READ_FAIL_THRESHOLD = 5


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
        subject_flush: Subject[FlushEvent],
        subject_new_video_start: Subject[NewVideoStartEvent],
        datetime_provider: DatetimeProvider,
        frame_counter: Counter,
        get_current_config: GetCurrentConfig,
        read_fail_threshold: int = DEFAULT_READ_FAIL_THRESHOLD,
    ) -> None:

        self.subject_flush = subject_flush
        self.subject_new_video_start = subject_new_video_start
        self._datetime_provider = datetime_provider
        self._stop_capture = False
        self._frame_counter = frame_counter
        self._get_current_config = get_current_config
        self._current_stream: str | None = None
        self._current_video_capture: VideoCapture | None = None
        self._stream_start_time: datetime = self._datetime_provider.provide()
        self._current_video_start_time = self._stream_start_time
        self._outdated = True
        self._read_fail_threshold = read_fail_threshold
        self._consecutive_read_fails = 0

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
        try:
            while not self.should_stop():
                if (frame := self._read_next_frame()) is not None:
                    self._frame_counter.increment()
                    occurrence = self._datetime_provider.provide()

                    if self._outdated:
                        self._current_video_start_time = occurrence
                        self._outdated = False
                        self._notify_new_video_start_observers()

                    yield Frame(
                        data=convert_frame_to_rgb(frame),  # YOLO expects RGB
                        frame=self.current_frame_number,
                        source=self.rtsp_url,
                        output=self.create_output(),
                        occurrence=occurrence,
                    )
                    if self.flush_condition_met():
                        self._notify_flush_observers()
                        self._outdated = True
                        self._frame_counter.reset()
            self._notify_flush_observers()
        except InvalidRtspUrlError as cause:
            logger().error(cause)

    def _init_video_capture(self, source: str) -> VideoCapture:
        self._wait_for_connection(source)

        cap = VideoCapture(source)
        while not self.should_stop() and not cap.isOpened():
            cap.release()
            self._wait_for_connection(source)
            cap = VideoCapture(source)
        return cap

    def _wait_for_connection(self, connection: str) -> None:
        while not self.should_stop() and not is_connection_available(connection):
            logger().debug(
                f"Couldn't open the RTSP stream: {connection}. "
                f"Trying again in {RETRY_SECONDS}s..."
            )
            sleep(RETRY_SECONDS)

    def _read_next_frame(self) -> ndarray | None:
        successful, frame = self._video_capture.read()
        if successful:
            self._consecutive_read_fails = 0
            return frame
        self._consecutive_read_fails += 1

        if self._consecutive_read_fails >= self._read_fail_threshold:
            self._try_reconnecting_stream()

        logger().debug("Failed to grab frame")
        return None

    def _try_reconnecting_stream(self) -> None:
        self._video_capture.release()
        self._current_video_capture = None
        if not self.should_stop() and self._current_stream is not None:
            self._current_video_capture = self._init_video_capture(self._current_stream)

    def should_stop(self) -> bool:
        return self._stop_capture

    def stop(self) -> None:
        self._stop_capture = True

    def start(self) -> None:
        self._stop_capture = False

    def flush_condition_met(self) -> bool:
        return self.current_frame_number % self.flush_buffer_size == 0

    def _notify_flush_observers(self) -> None:
        frame_width = self._get_width()
        frame_height = self._get_height()
        frames = (
            self.flush_buffer_size
            if self.current_frame_number % self.flush_buffer_size == 0
            else self.current_frame_number % self.flush_buffer_size
        )
        duration = timedelta(seconds=round(frames / self.fps))
        output = self.create_output()
        self.subject_flush.notify(
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

    def _get_width(self) -> int:
        return int(self._video_capture.get(CAP_PROP_FRAME_WIDTH))

    def _get_height(self) -> int:
        return int(self._video_capture.get(CAP_PROP_FRAME_HEIGHT))

    def _notify_new_video_start_observers(self) -> None:
        event = NewVideoStartEvent(
            output=self.create_output(),
            width=self._get_width(),
            height=self._get_height(),
            fps=self.fps,
        )
        self.subject_new_video_start.notify(event)

    def create_output(self) -> str:
        output_filename = (
            f"{self.stream_config.name}_FR{round(self.fps)}"
            f"_{self._current_video_start_time.strftime(DATETIME_FORMAT)}.mp4"
        )
        return str(self.stream_config.save_dir / output_filename)


def convert_frame_to_rgb(frame: ndarray) -> ndarray:
    return cvtColor(frame, COLOR_BGR2RGB)


class InvalidRtspUrlError(Exception):
    """Raised when the RTSP URL is invalid."""


def is_connection_available(rtsp_url: str) -> bool:
    """
    Check if RTSP connection is available by sending a DESCRIBE request.

    Args:
        rtsp_url: The RTSP URL to check

    Returns:
        bool: True if stream is available, False otherwise
    """
    try:
        parsed = urlparse(rtsp_url)
        if parsed.hostname is None and parsed.port is None:
            raise InvalidRtspUrlError(
                f"Invalid RTSP URL: {rtsp_url}. Missing hostname or port."
            )

        host = parsed.hostname
        port = parsed.port

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)

        if sock.connect_ex((host, port)) != 0:
            sock.close()
            return False

        # Send RTSP DESCRIBE request to get stream info
        rtsp_request = (
            f"DESCRIBE {rtsp_url} RTSP/1.0\r\n"
            f"CSeq: 1\r\n"
            f"Accept: application/sdp\r\n\r\n"
        )
        sock.send(rtsp_request.encode())

        # Read response
        response = sock.recv(4096).decode()
        sock.close()

        # Check if we got a valid RTSP response with SDP content
        return (
            response.startswith("RTSP/1.0 200 OK")
            and "application/sdp" in response
            and "m=video" in response
        )
    except InvalidRtspUrlError:
        raise
    except Exception:
        return False
