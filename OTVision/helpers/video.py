from datetime import timedelta
from pathlib import Path

import av
from av.container import InputContainer
from av.video.stream import VideoStream


def get_video_dimensions(video: Path) -> tuple[int, int]:
    """Get video width and height.

    Args:
        video (Path): the video file

    Returns:
        tuple[int, int]: width and height of video
    """
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
        first_frame = next(container.decode(stream))
        if abs(first_frame.rotation) in (90, 270):
            return height, width
        return width, height


def get_fps(video: Path) -> float:
    """Get video's fps.

    Args:
        video (Path): the video file

    Returns:
        float: the video's fps
    """
    with av.open(str(video)) as container:
        return _get_fps(container.streams.video[0])


def get_duration(video_file: Path) -> timedelta:
    """Get the duration of the video
    Args:
        video_file (Path): path to video file
    Returns:
        timedelta: duration of the video
    """
    with av.open(str(video_file.absolute())) as container:
        return timedelta(seconds=_get_duration_in_seconds(container))


def get_number_of_frames(video_file: Path) -> int:
    """Get the number of frames of the video
    Args:
        video_file (Path): path to video file
    Returns:
        timedelta: number of frames of the video
    """
    with av.open(str(video_file.absolute())) as container:
        fps = _get_fps(container.streams.video[0])
        return int(_get_duration_in_seconds(container) * fps)


def _get_fps(stream: VideoStream) -> float:
    rate = stream.average_rate or stream.base_rate
    if rate is None:
        raise ValueError(f"Could not determine fps of video stream {stream}")
    return float(rate)


def _get_duration_in_seconds(container: InputContainer) -> float:
    if container.duration is None:
        raise ValueError(f"Could not determine duration of video {container.name}")
    return container.duration / av.time_base


def convert_seconds_to_frames(seconds: int | None, fps: float) -> int | None:
    if seconds is None:
        return None
    return round(seconds * fps)
