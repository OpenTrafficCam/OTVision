import cv2
import re
import os
from pathlib import Path

FFMPEG_PATH = Path(__file__).parents[0] / r"ffmpeg.exe"


def convert(input_video, output_extension=".avi", output_fps=None, overwrite=True):
    input_path = Path(input_video)
    input_filename = input_path.stem
    output_path = input_path.with_suffix(output_extension)
    if not overwrite and output_path.is_file:
        return None
    vid_extensions = [
        ".mov",
        ".avi",
        ".mp4",
        ".mpg",
        ".mpeg",
        ".m4v",
        ".wmv",
        ".mkv",
        ".h264",
    ]
    try:
        input_fps = float(re.search("_FR(.*?)_", input_filename).group(1))
    except AttributeError:
        try:
            input_fps = cv2.VideoCapture(input_video).get(cv2.CAP_PROP_FPS)
        except:
            input_fps = None
    if output_fps is None:
        if input_fps is None:
            output_fps = 25.0
        else:
            output_fps = input_fps

    print(f"input fps: {input_fps}")
    print(f"output fps: {output_fps}")

    if input_path.suffix in vid_extensions and output_path.suffix in vid_extensions:
        ffmpeg_cmd = f"{FFMPEG_PATH} -r {str(output_fps)} -i {input_path} -vcodec copy -an -y {output_path}"
        os.system(ffmpeg_cmd)


if __name__ == "__main__":
    # test_video = str(Path(__file__).parents[2] / r"tests/data/testvideo_1.mkv")
    test_video = str(
        Path(__file__).parents[2]
        / r"tests/data/testvideo_FR20_2020-02-20_12-00-00.h264"
    )
    convert(input_video=test_video, output_extension=".avi")
