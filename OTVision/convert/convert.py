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


import subprocess
from pathlib import Path

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files
from OTVision.helpers.formats import _get_fps_from_filename
from OTVision.helpers.log import log, reset_debug, set_debug


def main(
    paths: list[Path],
    output_filetype: str = CONFIG["CONVERT"]["OUTPUT_FILETYPE"],
    input_fps: float = CONFIG["CONVERT"]["INPUT_FPS"],
    output_fps: float = CONFIG["CONVERT"]["OUTPUT_FPS"],
    fps_from_filename: bool = CONFIG["CONVERT"]["FPS_FROM_FILENAME"],
    overwrite: bool = CONFIG["CONVERT"]["OVERWRITE"],
    delete_input: bool = False,
    debug: bool = CONFIG["CONVERT"]["DEBUG"],
):
    """Converts multiple h264-based videos into other formats and/or frame rates.

    Currently only works for windows as ffmpeg.exe is utilized.

    Args:
        paths (list[Path]): List of paths to .h264 files
            (or other video files)
        output_filetype (str, optional): Type of video file created.
            Defaults to CONFIG["CONVERT"]["OUTPUT_FILETYPE"].
        input_fps (float, optional): Frame rate of input video.
            Defaults to CONFIG["CONVERT"]["INPUT_FPS"].
        output_fps (float, optional): Frame rate of output video.
            Defaults to CONFIG["CONVERT"]["OUTPUT_FPS"].
        fps_from_filename (bool, optional): Whether or not trying to parse frame rate
            from file name. Defaults to CONFIG["CONVERT"]["FPS_FROM_FILENAME"].
        overwrite (bool, optional): Whether or not to overwrite existing video files.
            Defaults to CONFIG["CONVERT"]["OVERWRITE"].
        debug (bool, optional): Whether or not logging in debug mode.
            Defaults to CONFIG["CONVERT"]["DEBUG"].
    """

    log.info("Start conversion")
    if debug:
        set_debug()

    check_ffmpeg()
    h264_files = get_files(paths, [".h264"])
    for h264_file in h264_files:
        convert(
            h264_file,
            output_filetype,
            input_fps,
            output_fps,
            fps_from_filename,
            overwrite,
            delete_input,
        )
    if debug:
        reset_debug()


def convert(
    input_video_file: Path,
    output_filetype: str = CONFIG["CONVERT"]["OUTPUT_FILETYPE"],
    input_fps: float = CONFIG["CONVERT"]["INPUT_FPS"],
    output_fps: float = CONFIG["CONVERT"]["OUTPUT_FPS"],
    fps_from_filename: bool = CONFIG["CONVERT"]["FPS_FROM_FILENAME"],
    overwrite: bool = CONFIG["CONVERT"]["OVERWRITE"],
    delete_input: bool = False,
    debug: bool = CONFIG["CONVERT"]["DEBUG"],
):
    """Converts h264-based videos into other formats and/or other frame rates.
    Also input frame rates can be given.
    If input video file is raw h264 and no input frame rate is given convert
    tries to parse frame rate from filename, otherwise sets default frame rate.

    Currently only works for windows as ffmpeg.exe is utilized.

    Args:
        input_video_file (Path): Path to h264 video file (or other format).
        output_filetype (str, optional): Type of video file created.
            Defaults to CONFIG["CONVERT"]["OUTPUT_FILETYPE"].
        input_fps (float, optional): Frame rate of input video.
            Defaults to CONFIG["CONVERT"]["INPUT_FPS"].
        output_fps (float, optional): Frame rate of output video.
            Defaults to CONFIG["CONVERT"]["OUTPUT_FPS"].
        fps_from_filename (bool, optional): Whether or not trying to parse frame rate
            from file name. Defaults to CONFIG["CONVERT"]["FPS_FROM_FILENAME"].
        overwrite (bool, optional): Whether or not to overwrite existing video files.
            Defaults to CONFIG["CONVERT"]["OVERWRITE"].
        delete_input (bool, optional): Whether or not to delete the input h264.
          Defaults to False.
        debug (bool, optional): Whether or not logging in debug mode.
            Defaults to CONFIG["CONVERT"]["DEBUG"].

    Raises:
        TypeError: If output video filetype is not supported.
        TypeError: If input video filetype is not supported.

    Returns:
        None: If not on a windows machine.
        None: If output video file already exists and overwrite is not enabled.
    """
    if debug:
        set_debug()

    log.info(f"Try converting {input_video_file} to {output_filetype}")

    input_filename = input_video_file.stem
    input_filetype = input_video_file.suffix
    output_video_file = input_video_file.with_suffix(output_filetype)
    if not overwrite and output_video_file.is_file():
        if debug:
            reset_debug()
        return None
    vid_filetypes = CONFIG["FILETYPES"]["VID"] + [".h264"]
    if input_filetype in vid_filetypes and output_filetype in vid_filetypes:
        if fps_from_filename:
            input_fps = _get_fps_from_filename(input_filename)
        elif input_fps is None:
            input_fps = CONFIG["CONVERT"]["FPS"]

        # Create ffmpeg command

        # Input frame rate
        # ? Change -framerate to -r?
        input_fps_cmds = ["-framerate", str(input_fps)] if input_fps is not None else []

        # Output frame rate and copy commands
        if output_fps is not None:
            output_fps_cmds = ["-r", str(output_fps)]
            copy_cmds: list[str] = []
            delete_input = False  # Never delete input if re-encoding file.
        else:
            output_fps_cmds = ""
            copy_cmds = ["-vcodec", "copy"]  # No re-encoding, only demuxing

        # Input file
        input_file_cmds = ["-i", str(input_video_file)]

        # Filters (mybe necessary for special cases, insert if needed)
        # ffmpeg_cmd_filter = "-c:v libx264"
        filter_cmds: list[str] = []

        # Output file
        output_file_cmds = ["-y", str(output_video_file)]

        # Concat and run ffmpeg command
        # ffmpeg_cmd = rf"ffmpeg {ffmpeg_cmd_in} {ffmpeg_cmd_out}"
        ffmpeg_cmd = (
            ["ffmpeg"]
            + input_fps_cmds
            + input_file_cmds
            + filter_cmds
            + output_fps_cmds
            + copy_cmds
            + output_file_cmds
        )
        log.debug(f"ffmpeg command: {ffmpeg_cmd}")

        subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        log.info(f"{output_video_file} created with {output_fps} fps")

        if delete_input:
            in_size = input_video_file.stat().st_size
            out_size = output_video_file.stat().st_size
            if in_size <= out_size:
                log.debug(f"Input file ({in_size}) <= output file ({out_size}).")
                input_video_file.unlink()

    elif input_filetype in vid_filetypes:
        raise TypeError("Output video filetype is not supported")
    elif output_filetype in vid_filetypes:
        raise TypeError("Input video filetype is not supported")

    if debug:
        reset_debug()


def check_ffmpeg():
    """Checks, if ffmpeg is available"""

    try:
        subprocess.call("ffmpeg")
        log.info("ffmpeg was found")
    except FileNotFoundError as e:
        error_message = "ffmpeg could not be called, make sure ffmpeg is in path"
        raise FileNotFoundError(error_message) from e


# Useful ffmpeg commands:
# "-vcodec"=Get video only
# "copy"=only demuxing and muxing
# "-bsf:v h264_mp4toannexb"
# "-c:v libx264"= encoder for h264 input stream
# "-bsf:v h264_mp4toannexb"=?
# "-y"=Overwrite output file
