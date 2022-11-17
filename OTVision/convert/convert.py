"""
OTVision main module for converting videos to other formats and frame rates.
"""
# Copyright (C) 2022 OpenTrafficCam Contributors
# <https://github.com/OpenTrafficCam
# <team@opentrafficcam.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import subprocess
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files, remove_dir
from OTVision.helpers.formats import _get_fps_from_filename
from OTVision.helpers.log import log
from OTVision.helpers.machine import ON_WINDOWS


def main(
    paths,
    output_filetype: str = CONFIG["CONVERT"]["OUTPUT_FILETYPE"],
    input_fps: float = None,
    output_fps: float = None,
    fps_from_filename: bool = True,
    overwrite: bool = CONFIG["CONVERT"]["OVERWRITE"],
    debug: bool = CONFIG["CONVERT"]["DEBUG"],
    # TODO: #111 Set more parameters as global variables in config.py
):
    """Converts multiple h264-based videos into other formats and/or other frame rates.

    Currently only works for windows as ffmpeg.exe is utilized.

    Args:
        paths ([type]): [description]
        output_filetype (str, optional): [description]. Defaults to None.
        input_fps (float, optional): [description]. Defaults to None.
        output_fps (float, optional): [description]. Defaults to None.
        overwrite (bool, optional): [description]. Defaults to True.
    """

    log.info("Start conversion")
    if debug:
        log.setLevel("DEBUG")
        log.debug("Debug mode on")

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
    input_video_file: str,
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

    Currently only works for windows as ffmpeg.exe is utilized.

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

    if not ON_WINDOWS:
        log.warning("Conversion of h264 videos only works on windows machines for now")
        return

    log.info(f"Try converting {input_video_file} to {output_filetype}")

    input_video_file = Path(input_video_file)
    input_filename = input_video_file.stem
    input_filetype = input_video_file.suffix
    output_video_file = input_video_file.with_suffix(output_filetype)
    if not overwrite and output_video_file.is_file():
        return None
    vid_filetypes = CONFIG["FILETYPES"]["VID"] + [".h264"]
    if input_filetype in vid_filetypes and output_filetype in vid_filetypes:
        if fps_from_filename:
            input_fps = _get_fps_from_filename(input_filename)
        elif input_fps is None:
            input_fps = CONFIG["CONVERT"]["FPS"]

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
        ffmpeg_cmd_in = f"{input_fps_cmd} -i {input_video_file}"
        # Filters (mybe necessary for special cases, insert )
        # ffmpeg_cmd_filter = "-c:v libx264"
        ffmpeg_cmd_filter = ""
        # Output file
        ffmpeg_cmd_out = (
            f"{ffmpeg_cmd_filter} {output_fps_cmd} {copy_cmd} -y {output_video_file}"
        )
        # Concat and run ffmpeg command
        FFMPEG_PATH = CONFIG["CONVERT"]["FFMPEG_PATH"]
        ffmpeg_cmd = rf"{FFMPEG_PATH} {ffmpeg_cmd_in} {ffmpeg_cmd_out}"
        log.debug(f"ffmpeg command: {ffmpeg_cmd}")

        subprocess.call(ffmpeg_cmd)
        log.info(f"{output_video_file} created with {output_fps} fps")

    elif input_filetype in vid_filetypes:
        raise TypeError("Output video filetype is not supported")
    elif output_filetype in vid_filetypes:
        raise TypeError("Input video filetype is not supported")


def check_ffmpeg():
    """Checks, if ffmpeg is available, otherwise downloads it.
    Args:
        ffmpeg_path (str): path, where to save ffmpeg
    """

    if not ON_WINDOWS:
        log.warning("Sorry, this function only works on windows machines for now")
        return

    try:
        subprocess.call(CONFIG["CONVERT"]["FFMPEG_PATH"])
        log.info("ffmpeg.exe was found")
    except FileNotFoundError:
        download_ffmpeg()


def download_ffmpeg():
    """Download ffmpeg to a specific path.
    Args:
        ffmpeg_path (str): path to ffmpeg.exe
    """

    if not ON_WINDOWS:
        log.info("Sorry, this function only works on windows machines for now")
        return

    log.info("Try downloading ffmpeg zip archive (patience: may take a while...)")
    FFMPEG_DIR = str(Path(CONFIG["CONVERT"]["FFMPEG_PATH"]).parents[0])
    if not Path(str(Path(FFMPEG_DIR) / "tmp")).is_dir():
        os.mkdir(str(Path(FFMPEG_DIR) / "tmp"))
    FFMPEG_ZIP = str(Path(FFMPEG_DIR) / "tmp" / r"ffmpeg.zip")
    FFMPEG_ZIP_DIR = str(Path(FFMPEG_ZIP).parents[0])
    try:
        urlretrieve(CONFIG["CONVERT"]["FFMPEG_URL"], FFMPEG_ZIP)
        log.info("Successfully downloaded ffmpeg zip archive")
    except Exception as inst:
        log.warning(inst)
        log.warning("Can't download ffmpeg zip archive. Please download manually")
    else:
        try:
            log.info("Extracting ffmpeg.exe from ffmpeg zip archive")
            with ZipFile(FFMPEG_ZIP, "r") as zip:
                for name in zip.namelist():
                    if Path(name).name == r"ffmpeg.exe":
                        log.info("next: Extract")
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
            log.info("Successfully extracted ffmpeg.exe from ffmpeg zip archive")
        except Exception as inst:
            log.warning(inst)
            log.warning("Can't extract ffmpeg.exe, please extract manual")


# Useful ffmpeg commands:
# "-vcodec"=Get video only
# "copy"=only demuxing and muxing
# "-bsf:v h264_mp4toannexb"
# "-c:v libx264"= encoder for h264 input stream
# "-bsf:v h264_mp4toannexb"=?
# "-y"=Overwrite output file
