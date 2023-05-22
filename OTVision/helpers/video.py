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
