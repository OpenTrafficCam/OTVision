import logging
from enum import IntEnum, StrEnum
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired
from threading import Thread
from typing import Callable, Generator

import ffmpeg
from numpy import ndarray

from OTVision.application.event.new_video_start import NewVideoStartEvent
from OTVision.detect.detected_frame_buffer import FlushEvent
from OTVision.domain.frame import Frame, FrameKeys
from OTVision.domain.video_writer import VideoWriter
from OTVision.helpers.log import LOGGER_NAME
from OTVision.helpers.machine import ON_WINDOWS

VideoSaveLocationStrategy = Callable[[str], str]

BUFFER_SIZE_100MB = 10**8
DEFAULT_CRF = 23
VIDEO_SAVE_FILE_POSTFIX = "_processed"

log = logging.getLogger(LOGGER_NAME)


class VideoCodec(StrEnum):
    """Enum of possible video codecs to be used with ffmpeg

    Attributes:
        H264_SOFTWARE: Software-based H.264 encoder.
        H264_NVENC: NVIDIA GPU-based H.264 encoder.
        H264_QSV: Intel Quick Sync Video encoder.
        H264_VAAPI: Intel/AMD GPU-based H.264 encoder.
        H264_VIDEOTOOLBOX: macOS hardware encoder.
    """

    H264_SOFTWARE = "libx264"
    H264_NVENC = "h264_nvenc"
    H264_QSV = "h264_qsv"
    H264_VAAPI = "h264_vaapi"
    H264_VIDEOTOOLBOX = "h264_videotoolbox"

    @staticmethod
    def as_list() -> list[str]:
        return list(VideoCodec.__members__.values())


class VideoFormat(StrEnum):
    RAW = "rawvideo"
    MP4 = "mp4"

    @staticmethod
    def as_list() -> list[str]:
        return list(VideoFormat.__members__.values())


class PixelFormat(StrEnum):
    YUV420P = "yuv420p"  # compatible with most players for H.264
    RGB24 = "rgb24"
    BGR24 = "bgr24"

    @staticmethod
    def as_list() -> list[str]:
        return list(PixelFormat.__members__.values())


class EncodingSpeed(StrEnum):
    ULTRA_FAST = "ultrafast"
    SUPER_FAST = "superfast"
    VERY_FAST = "veryfast"
    FASTER = "faster"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    SLOWER = "slower"
    VERY_SLOW = "veryslow"

    @staticmethod
    def as_list() -> list[str]:
        return list(EncodingSpeed.__members__.values())


class ConstantRateFactor(IntEnum):
    """Adjust quality/size (lower means better quality/larger file).
    Attributes:
        LOSSLESS: Perfect quality, massive file size.
        HIGH_QUALITY: Visually lossless for most eyes.
        GOOD: High quality, slightly compressed.
        DEFAULT: x264 default; good balance of size and quality.
        COMPACT: Acceptable quality for small screens or streaming.
        LOW_QUALITY: Noticeable compression artifacts; smaller file.
        WORST_ACCEPTABLE: Very low quality; only for previews or constrained storage.
    """

    LOSSLESS = 0
    HIGH_QUALITY = 18
    GOOD = 20
    DEFAULT = 23
    COMPACT = 26
    LOW_QUALITY = 28
    WORST_ACCEPTABLE = 35

    @staticmethod
    def as_list() -> list[str]:
        return list(ConstantRateFactor.__members__.keys())


class FfmpegVideoWriter(VideoWriter):

    @property
    def _current_video_metadata(self) -> NewVideoStartEvent:
        if self.__current_video_metadata is None:
            raise ValueError("FfmpegVideoWriter is not configured yet.")
        return self.__current_video_metadata

    @property
    def _ffmpeg_process(self) -> Popen:
        if self.__ffmpeg_process is None:
            raise ValueError("FfmpegVideoWriter is not initialized yet.")
        return self.__ffmpeg_process

    @property
    def is_open(self) -> bool:
        return self.__ffmpeg_process is not None

    @property
    def is_closed(self) -> bool:
        return self.is_open is False

    def __init__(
        self,
        save_location_strategy: VideoSaveLocationStrategy,
        encoding_speed: EncodingSpeed = EncodingSpeed.FAST,
        input_format: VideoFormat = VideoFormat.RAW,
        output_format: VideoFormat = VideoFormat.MP4,
        input_pixel_format: PixelFormat = PixelFormat.RGB24,
        output_pixel_format: PixelFormat = PixelFormat.YUV420P,
        output_video_codec: VideoCodec = VideoCodec.H264_SOFTWARE,
        constant_rate_factor: ConstantRateFactor = ConstantRateFactor.LOSSLESS,
    ) -> None:
        if ON_WINDOWS:
            log.warning(
                "Writing every frame into a new video is not supported on Windows."
            )
        self._save_location_strategy = save_location_strategy
        self._encoding_speed = encoding_speed
        self._input_format = input_format
        self._output_format = output_format
        self._input_pixel_format = input_pixel_format
        self._output_pixel_format = output_pixel_format
        self._output_video_codec = output_video_codec
        self.__ffmpeg_process: Popen | None = None
        self.__current_video_metadata: NewVideoStartEvent | None = None
        self._constant_rate_factor = constant_rate_factor
        log.info(
            "FFmpeg video writer settings: "
            f"video_codec='{self._output_video_codec.value}', "
            f"encoding_speed='{self._encoding_speed.value}', "
            f"crf='{self._constant_rate_factor.value}'"
        )

    def open(self, output: str, width: int, height: int, fps: float) -> None:
        self.__ffmpeg_process = self.__create_ffmpeg_process(
            output_file=output,
            width=width,
            height=height,
            fps=fps,
        )

    def write(self, image: ndarray) -> None:
        try:
            if self._ffmpeg_process.stdin:
                self._ffmpeg_process.stdin.write(image.tobytes())
                self._ffmpeg_process.stdin.flush()
        except BrokenPipeError:
            # Check if the process is still running
            if self._ffmpeg_process.poll() is not None:
                # Process has terminated, get the error message
                stderr = (
                    self._ffmpeg_process.stderr.read()
                    if self._ffmpeg_process.stderr
                    else b""
                )
                log.info(stderr.decode("utf-8", errors="ignore"))
                raise RuntimeError(
                    "ffmpeg process terminated unexpectedly: "
                    f"{stderr.decode('utf-8', errors='ignore')}"
                )
            raise  # Re-raise the original exception if the process is still running

    def close(self) -> None:
        if self.__ffmpeg_process is not None:
            process_to_cleanup = self._ffmpeg_process
            self.__ffmpeg_process = None  # Immediately mark as closed

            # Close stdin synchronously (fast operation)
            try:
                if process_to_cleanup.stdin:
                    process_to_cleanup.stdin.flush()
                    process_to_cleanup.stdin.close()
            except Exception as cause:
                log.debug(f"Error closing stdin: {cause}")

            # Handle cleanup in background thread
            def cleanup_process() -> None:
                try:
                    # Just check if it's still running and wait for it
                    if process_to_cleanup.poll() is None:
                        # Still running, close stdin and wait
                        try:
                            if (
                                process_to_cleanup.stdin
                                and not process_to_cleanup.stdin.closed
                            ):
                                process_to_cleanup.stdin.close()
                        except Exception as cause:
                            log.error(f"Error closing stdin: {cause}")

                        # Simple wait with timeout
                        try:
                            process_to_cleanup.wait(timeout=5.0)
                        except TimeoutExpired:
                            process_to_cleanup.kill()
                            try:
                                process_to_cleanup.wait(timeout=1.0)
                            except TimeoutExpired:
                                log.error("Could not kill FFmpeg process")

                    # Log return code if we care
                    if process_to_cleanup.returncode != 0:
                        log.debug(
                            f"FFmpeg ended with code: {process_to_cleanup.returncode}"
                        )

                except Exception as e:
                    log.debug(f"Cleanup completed with minor issues: {e}")

            # Start cleanup in daemon thread (won't prevent program exit)
            cleanup_thread = Thread(target=cleanup_process, daemon=True)
            cleanup_thread.start()

        self.__current_video_metadata = None

    def notify_on_flush_event(self, event: FlushEvent) -> None:
        self.close()

    def notify_on_new_video_start(self, event: NewVideoStartEvent) -> None:
        self.__current_video_metadata = event
        self.open(event.output, event.width, event.height, event.fps)

    def __create_ffmpeg_process(
        self, output_file: str, width: int, height: int, fps: float
    ) -> Popen:
        save_file = self._save_location_strategy(output_file)
        cmd = (
            ffmpeg.input(
                "pipe:0",
                format=self._input_format.value,
                framerate=fps,
                pix_fmt=self._input_pixel_format.value,
                s=f"{width}x{height}",
            )
            .output(
                save_file,
                pix_fmt=self._output_pixel_format.value,
                vcodec=self._output_video_codec.value,
                preset=self._encoding_speed.value,
                crf=self._constant_rate_factor.value,
                format=self._output_format.value,
            )
            .overwrite_output()
            .compile()
        )

        process = Popen(
            cmd,
            stdin=PIPE,
            stderr=PIPE,
            bufsize=BUFFER_SIZE_100MB,
        )
        log.info(f"Writing new video file to '{save_file}'.")
        return process

    def filter(
        self, pipe: Generator[Frame, None, None]
    ) -> Generator[Frame, None, None]:
        for frame in pipe:
            if (image := frame.get(FrameKeys.data)) is not None:
                self.write(image)
            yield frame


def append_save_suffix_to_save_location(given: str) -> str:
    filepath = Path(given)
    return str(Path(filepath).with_stem(f"{filepath.stem}{VIDEO_SAVE_FILE_POSTFIX}"))


def keep_original_save_location(given: str) -> str:
    return given
