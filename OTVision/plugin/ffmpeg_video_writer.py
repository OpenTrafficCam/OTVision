from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any, Generator

import ffmpeg
from numpy import ndarray

from OTVision.abstraction.pipes_and_filter import Filter
from OTVision.detect.detected_frame_buffer import FlushEvent
from OTVision.domain.frame import Frame, FrameKeys
from OTVision.domain.video_writer import VideoWriter

DEFAULT_CRF = 23


class VideoCodec(StrEnum):
    H264 = "libx264"


class VideoFormat(StrEnum):
    RAW = "rawvideo"
    MP4 = "mp4"


class PixelFormat(StrEnum):
    YUV420P = "yuv420p"  # compatible with most players for H.264
    RGB24 = "rgb24"  # adjust quality/size (lower means better quality/larger file)
    BGR24 = "bgr24"


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


class ConstantRateFactor(IntEnum):
    LOSSLESS = 0
    DEFAULT = 23


@dataclass(frozen=True)
class NewVideoStartEvent:
    output: str
    width: int
    height: int
    fps: float


class FfmpegVideoWriter(VideoWriter, Filter[Frame, Frame]):

    @property
    def _current_video_metadata(self) -> NewVideoStartEvent:
        if self.__current_video_metadata is None:
            raise ValueError("FfmpegVideoWriter is not configured yet.")
        return self.__current_video_metadata

    @property
    def _ffmpeg_process(self) -> Any:
        if self.__ffmpeg_process is None:
            raise ValueError("FfmpegVideoWriter is not initialized yet.")
        return self.__ffmpeg_process

    def __init__(
        self,
        encoding_speed: EncodingSpeed = EncodingSpeed.FAST,
        input_format: VideoFormat = VideoFormat.RAW,
        output_format: VideoFormat = VideoFormat.MP4,
        input_pixel_format: PixelFormat = PixelFormat.RGB24,
        output_pixel_format: PixelFormat = PixelFormat.YUV420P,
        output_video_codec: VideoCodec = VideoCodec.H264,
        constant_rate_factor: ConstantRateFactor = ConstantRateFactor.LOSSLESS,
    ) -> None:
        self._encoding_speed = encoding_speed
        self._input_format = input_format
        self._output_format = output_format
        self._input_pixel_format = input_pixel_format
        self._output_pixel_format = output_pixel_format
        self._output_video_codec = output_video_codec
        self.__ffmpeg_process = None
        self.__current_video_metadata: NewVideoStartEvent | None = None
        self._constant_rate_factor = constant_rate_factor

    def open(self, output: str, width: int, height: int, fps: float) -> None:
        self.__ffmpeg_process = self.__create_ffmpeg_process(
            output_file=output,
            width=width,
            height=height,
            fps=fps,
        )

    def write(self, image: ndarray) -> None:
        try:
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
                raise RuntimeError(
                    "ffmpeg process terminated unexpectedly: "
                    f"{stderr.decode('utf-8', errors='ignore')}"
                )
            raise  # Re-raise the original exception if the process is still running

    def close(self) -> None:
        if self.__ffmpeg_process is not None:
            self._ffmpeg_process.stdin.flush()
            self._ffmpeg_process.stdin.close()
            self._ffmpeg_process.wait()

    def notify_on_flush_event(self, event: FlushEvent) -> None:
        self.close()

    def notify_on_new_video_start(self, event: NewVideoStartEvent) -> None:
        self.open(event.output, event.width, event.height, event.fps)

    def __create_ffmpeg_process(
        self, output_file: str, width: int, height: int, fps: float
    ) -> Any:
        process = (
            ffmpeg.input(
                "pipe:0",
                format=self._input_format.value,
                framerate=fps,
                pix_fmt=self._input_pixel_format.value,
                s=f"{width}x{height}",
            )
            .output(
                output_file,
                pix_fmt=self._output_pixel_format.value,
                vcodec=self._output_video_codec.value,
                preset=self._encoding_speed.value,
                crf=self._constant_rate_factor.value,
                format=self._output_format.value,
            )
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True)
        )
        return process

    def filter(
        self, pipe: Generator[Frame, None, None]
    ) -> Generator[Frame, None, None]:
        for frame in pipe:
            if (image := frame.get(FrameKeys.data)) is not None:
                self.write(image)
            yield frame
