import os
import re
import subprocess
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files, remove_dir


def main(
    paths,
    output_filetype: str = CONFIG["CONVERT"]["OUTPUT_FILETYPE"],
    input_fps: float = None,
    output_fps: float = None,
    fps_from_filename: bool = True,
    overwrite: bool = True,
    # TODO: #111 Set more parameters as global variables in config.py
):
    """Converts multiple h264-based videos into other formats and/or other frame rates.

    Args:
        paths ([type]): [description]
        output_filetype (str, optional): [description]. Defaults to None.
        input_fps (float, optional): [description]. Defaults to None.
        output_fps (float, optional): [description]. Defaults to None.
        overwrite (bool, optional): [description]. Defaults to True.
    """

    check_ffmpeg()
    h264_files = get_files(paths, ".h264")
    for h264_file in h264_files:
        convert(
            h264_file,
            output_filetype,
            input_fps,
            output_fps,
            fps_from_filename,
            overwrite,
        )


def convert(
    input_video: str,
    output_filetype: str = CONFIG["CONVERT"]["OUTPUT_FILETYPE"],
    input_fps: float = None,
    output_fps: float = None,
    fps_from_filename: bool = True,
    overwrite: bool = True,
):  # sourcery skip: inline-variable, simplify-fstring-formatting
    """Converts h264-based videos into other formats and/or other frame rates. Also
    input frame rates can be given. If input video file is raw h264 and no input frame
    rate is given it tries to read "FR" from filename and otherwise sets default frame
    rate of 25.

    Args:
        input_video (str): [description]
        output_filetype (str, optional): [description]. Defaults to ".avi".
        input_fps (float, optional): [description]. Defaults to None.
        output_fps (float, optional): [description]. Defaults to None.
        overwrite (bool, optional): [description]. Defaults to True.

    Raises:
        TypeError: [description]
        TypeError: [description]

    Returns:
        [type]: [description]
    """

    print(f"Preparing {input_video} for conversion")

    input_path = Path(input_video)
    input_filename = input_path.stem
    input_filetype = input_path.suffix
    output_path = input_path.with_suffix(output_filetype)
    if not overwrite and output_path.is_file:
        return None
    vid_filetypes = CONFIG["FILETYPES"]["VID"] + [".h264"]
    if input_filetype in vid_filetypes and output_filetype in vid_filetypes:
        if fps_from_filename:
            try:
                # Get input fps frome filename  #TODO: Check regex for numbers
                input_fps = float(re.search("_FR(.*?)_", input_filename).group(1))
            except AttributeError("Frame rate not found in filename"):
                pass
        elif input_fps is None:
            input_fps = CONFIG["CONVERT"]["FPS"]

        print("Creating ffmpeg command")

        # Create ffmpeg command
        input_fps_cmd = (
            f"-framerate {input_fps}" if input_fps is not None else ""
        )  # ? Change -framerate to -r?
        if output_fps is not None:
            output_fps_cmd = f"-r {output_fps}"
            copy_cmd = ""
        else:
            output_fps_cmd = ""
            copy_cmd = "-vcodec copy"  # No decoding, only demuxing
        # Input file
        ffmpeg_cmd_in = f"{input_fps_cmd} -i {input_path}"
        # Filters (mybe necessary for special cases, insert )
        # ffmpeg_cmd_filter = "-c:v libx264"
        ffmpeg_cmd_filter = ""
        # Output file
        ffmpeg_cmd_out = (
            f"{ffmpeg_cmd_filter} {output_fps_cmd} {copy_cmd} -y {output_path}"
        )
        # Concat and run ffmpeg command
        FFMPEG_PATH = CONFIG["CONVERT"]["FFMPEG_PATH"]
        ffmpeg_cmd = rf"{FFMPEG_PATH} {ffmpeg_cmd_in} {ffmpeg_cmd_out}"
        print(f"ffmpeg command: {ffmpeg_cmd}")

        print(f"Converting from {input_fps} fps to {output_fps} fps {output_filetype}")
        subprocess.call(ffmpeg_cmd)
        print(f"Converted successfully to {output_path}")

    elif input_filetype in vid_filetypes:
        raise TypeError("Output video filetype is not supported")
    elif output_filetype in vid_filetypes:
        raise TypeError("Input video filetype is not supported")


def check_ffmpeg():
    """Checks, if ffmpeg is available, otherwise downloads it.
    Args:
        ffmpeg_path (str): path, where to save ffmpeg
    """
    try:
        subprocess.call(CONFIG["CONVERT"]["FFMPEG_PATH"])
    except FileNotFoundError:
        download_ffmpeg()


def download_ffmpeg():
    """Download ffmpeg to a specific path.
    Args:
        ffmpeg_path (str): path to ffmpeg.exe
    """
    print("Downloading ffmpeg zip archive (patience: may take a while...)")
    FFMPEG_DIR = str(Path(CONFIG["CONVERT"]["FFMPEG_PATH"]).parents[0])
    if not Path(str(Path(FFMPEG_DIR) / "tmp")).is_dir():
        os.mkdir(str(Path(FFMPEG_DIR) / "tmp"))
    FFMPEG_ZIP = str(Path(FFMPEG_DIR) / "tmp" / r"ffmpeg.zip")
    FFMPEG_ZIP_DIR = str(Path(FFMPEG_ZIP).parents[0])
    try:
        urlretrieve(CONFIG["CONVERT"]["FFMPEG_URL"], FFMPEG_ZIP)
        print("Successfully downloaded ffmpeg zip archive")
    except Exception as inst:
        print(inst)
        print("Can't download ffmpeg zip archive. Please download manually")
    else:
        try:
            print("Extracting ffmpeg.exe from ffmpeg zip archive")
            with ZipFile(FFMPEG_ZIP, "r") as zip:
                for name in zip.namelist():
                    if Path(name).name == r"ffmpeg.exe":
                        print("next: Extract")
                        zip.extract(
                            member=name,
                            path=FFMPEG_ZIP_DIR,
                        )
                        ffmpeg_exe = Path(name)
                        break
            os.rename(
                str(Path(FFMPEG_ZIP_DIR) / str(ffmpeg_exe)),
                str(Path(FFMPEG_DIR) / "ffmpeg.exe"),
            )
            remove_dir(dir=FFMPEG_ZIP_DIR)
            print("Successfully extracted ffmpeg.exe from ffmpeg zip archive")
        except Exception as inst:
            print(inst)
            print("Can't extract ffmpeg.exe, please extract manual")


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
