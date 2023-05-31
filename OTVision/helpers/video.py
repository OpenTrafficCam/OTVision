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
    video_clip = VideoFileClip(str(video))
    video_dimensions = video_clip.size
    video_clip.close()

    return video_dimensions


def get_fps(video: str) -> float:
    """Get video's fps.

    Args:
        video (Path): the video file

    Returns:
        float: the video's fps
    """
    video_clip = VideoFileClip(str(video))
    fps = video_clip.fps
    video_clip.close()

    return fps


def get_duration(video_file: Path) -> timedelta:
    """Get the duration of the video
    Args:
        video_file (Path): path to video file
    Returns:
        timedelta: duration of the video
    """
    clip = VideoFileClip(str(video_file.absolute()))
    return timedelta(seconds=clip.duration)


def get_number_of_frames(video_file: Path) -> int:
    """Get the number of frames of the video
    Args:
        video_file (Path): path to video file
    Returns:
        timedelta: number of frames of the video
    """
    clip = VideoFileClip(str(video_file.absolute()))
    return clip.reader.nframes
