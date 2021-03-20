import re
import os
from pathlib import Path


def convert(
    input_video: str,
    output_filetype: str = ".avi",
    input_fps: float = None,
    output_fps: float = None,
    overwrite: bool = True,
):
    FFMPEG_PATH = Path(__file__).parents[0] / r"ffmpeg.exe"
    DEFAULT_FPS = 25.0
    input_path = Path(input_video)
    input_filename = input_path.stem
    input_filetype = input_path.suffix
    output_path = input_path.with_suffix(output_filetype)
    if not overwrite and output_path.is_file:
        return None
    vid_filetypes = [
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
    if input_filetype in vid_filetypes and output_filetype in vid_filetypes:
        if input_filetype == ".h264" and input_fps is None:
            try:
                # Get input fps frome filename
                input_fps = float(re.search("_FR(.*?)_", input_filename).group(1))
            except AttributeError("Frame rate not found in filename"):
                input_fps = DEFAULT_FPS

        print(f"Input fps: {input_fps}")
        print(f"Output fps: {output_fps}")

        # Create ffmpeg command
        if input_fps is not None:
            input_fps_cmd = f"-framerate {input_fps}"
        else:
            input_fps_cmd = ""
        if output_fps is not None:
            output_fps_cmd = f"-r {output_fps}"
            copy_cmd = ""
        else:
            output_fps_cmd = ""
            copy_cmd = "-vcodec copy"  # No decoding, only demuxing
        # Input file
        ffmpeg_cmd_in = f"{input_fps_cmd} -i {input_path}"
        # Filters
        ffmpeg_cmd_filter = f"-c:v libx264 {output_fps_cmd} {copy_cmd}"
        # Output file
        ffmpeg_cmd_out = f"-y {output_path}"
        # Concat and run ffmpeg command
        ffmpeg_cmd = (
            f"{FFMPEG_PATH} {ffmpeg_cmd_in} {ffmpeg_cmd_filter} {ffmpeg_cmd_out}"
        )
        os.system(ffmpeg_cmd)

    elif input_filetype in vid_filetypes:
        raise TypeError("Output video filetype is not supported")
    elif output_filetype in vid_filetypes:
        raise TypeError("Input video filetype is not supported")


if __name__ == "__main__":
    # test_video = str(Path(__file__).parents[2] / r"tests/data/testvideo_1.mkv")
    test_video = str(
        Path(__file__).parents[2]
        / r"tests/data/testvideo_FR20_2020-02-20_12-00-00.h264"
    )
    convert(input_video=test_video, output_filetype=".avi")

# Useful ffmpeg commands:
# "-vcodec"=Get video only
# "copy"=only demuxing and muxing
# "-bsf:v h264_mp4toannexb"
# "-c:v libx264"= encoder for h264 input stream
# "-bsf:v h264_mp4toannexb"=?
# "-y"=Overwrite output file
