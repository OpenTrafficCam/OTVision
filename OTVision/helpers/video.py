from datetime import timedelta
from pathlib import Path

from moviepy.video.io.VideoFileClip import VideoFileClip


def get_video_dimensions(video: Path) -> tuple[int, int]:
    """Get video width and height.

    Args:
        video (Path): the video file

    Returns:
        tuple[int, int]: width and height of video
    """
    with VideoFileClip(str(video)) as clip:
        video_dimensions = clip.size
        return video_dimensions


def get_fps(video: Path) -> float:
    """Get video's fps.

    Args:
        video (Path): the video file

    Returns:
        float: the video's fps
    """
    with VideoFileClip(str(video)) as clip:
        fps = clip.fps
        return fps


def get_duration(video_file: Path) -> timedelta:
    """Get the duration of the video
    Args:
        video_file (Path): path to video file
    Returns:
        timedelta: duration of the video
    """
    with VideoFileClip(str(video_file.absolute())) as clip:
        return timedelta(seconds=clip.duration)


def get_number_of_frames(video_file: Path) -> int:
    """Get the number of frames of the video
    Args:
        video_file (Path): path to video file
    Returns:
        timedelta: number of frames of the video
    """
    with VideoFileClip(str(video_file.absolute())) as clip:
        return clip.reader.nframes


def convert_seconds_to_frames(seconds: int | None, fps: float) -> int | None:
    if seconds is None:
        return None
    return round(seconds * fps)
