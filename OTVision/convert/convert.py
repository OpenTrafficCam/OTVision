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


import logging
import subprocess
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from OTVision.config import (
    CONFIG,
    CONVERT,
    DELETE_INPUT,
    FILETYPES,
    FPS_FROM_FILENAME,
    INPUT_FPS,
    OUTPUT_FILETYPE,
    OVERWRITE,
    ROTATION,
    VID,
    VID_ROTATABLE,
)
from OTVision.helpers.files import get_files
from OTVision.helpers.formats import _get_fps_from_filename
from OTVision.helpers.log import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)

OUTPUT_FPS: Optional[float] = None
CONVERTABLE_FILETYPES = list(
    set(CONFIG[FILETYPES][VID]).union([".h264"]).difference([".mp4"])
)


def main(
    paths: list[Path],
    output_filetype: str = CONFIG[CONVERT][OUTPUT_FILETYPE],
    input_fps: float = CONFIG[CONVERT][INPUT_FPS],
    fps_from_filename: bool = CONFIG[CONVERT][FPS_FROM_FILENAME],
    rotation: int = CONFIG[CONVERT][ROTATION],
    overwrite: bool = CONFIG[CONVERT][OVERWRITE],
    delete_input: bool = CONFIG[CONVERT][DELETE_INPUT],
) -> None:
    """Converts multiple h264-based videos into other formats.

    Args:
        paths (list[Path]): List of paths to .h264 files
            (or other video files)
        output_filetype (str, optional): Extension and format of video file created.
            Defaults to CONFIG["CONVERT"]["OUTPUT_FILETYPE"].
        input_fps (float, optional): Frame rate of input h264.
            If fps_from_filename is set to True, input_fps will be ignored.
            Defaults to CONFIG["CONVERT"]["INPUT_FPS"].
        fps_from_filename (bool, optional): Whether to parse frame rate
            from file name. Defaults to CONFIG["CONVERT"]["FPS_FROM_FILENAME"].
        rotation (int, optional): Add rotation information to video metadata.
            Defaults to CONFIG["CONVERT"]["ROTATION"].
        overwrite (bool, optional): Whether to overwrite existing video files.
            Defaults to CONFIG["CONVERT"]["OVERWRITE"].
        delete_input (bool, optional): Whether to delete the input h264.
            Defaults to CONFIG["CONVERT"]["DELETE_INPUT"].
    """
    files = get_files(paths, CONVERTABLE_FILETYPES)

    start_msg = f"Start conversion of {len(files)} files"
    log.info(start_msg)
    print(start_msg)

    if not files:
        log.warning("No files found to convert!")
        return

    check_ffmpeg()

    for _file in tqdm(files, desc="Converted files", unit="files"):
        log.info(f"Convert {_file} to {output_filetype}")
        convert(
            _file,
            output_filetype,
            input_fps,
            fps_from_filename,
            rotation,
            overwrite,
            delete_input,
        )
        log.info(f"Successfully converted {_file} to {output_filetype}")

    finished_msg = "Finished conversion"
    log.info(finished_msg)
    print(finished_msg)


def convert(
    input_video_file: Path,
    output_filetype: str = CONFIG[CONVERT][OUTPUT_FILETYPE],
    input_fps: float = CONFIG[CONVERT][INPUT_FPS],
    fps_from_filename: bool = CONFIG[CONVERT][FPS_FROM_FILENAME],
    rotation: int = CONFIG[CONVERT][ROTATION],
    overwrite: bool = CONFIG[CONVERT][OVERWRITE],
    delete_input: bool = CONFIG[CONVERT][DELETE_INPUT],
) -> None:
    """Converts h264-based videos into other formats and/or other frame rates.
    Also input frame rates can be given.
    If input video file is raw h264 and no input frame rate is given convert
    tries to parse frame rate from filename, otherwise sets default frame rate.

    Currently only works for windows as ffmpeg.exe is utilized.

    Args:
        input_video_file (Path): Path to h264 video file (or other format).
        output_filetype (str, optional): Type of video file created.
            Defaults to CONFIG["CONVERT"]["OUTPUT_FILETYPE"].
        input_fps (float, optional): Frame rate of input h264.
            If fps_from_filename is set to True, input_fps will be ignored.
            Defaults to CONFIG["CONVERT"]["INPUT_FPS"].
        fps_from_filename (bool, optional): Whether to parse frame rate
            from file name. Defaults to CONFIG["CONVERT"]["FPS_FROM_FILENAME"].
        rotation (int, optional): Add rotation information to video metadata.
            Defaults to CONFIG["CONVERT"]["ROTATION"].
        overwrite (bool, optional): Whether to overwrite existing video files.
            Defaults to CONFIG["CONVERT"]["OVERWRITE"].
        delete_input (bool, optional): Whether to delete the input h264.
          Defaults to CONFIG["CONVERT"]["DELETE_INPUT"].

    Raises:
        TypeError: If output video filetype is not supported.
        TypeError: If input video filetype is not supported.

    Returns:
        None: If not on a windows machine.
        None: If output video file already exists and overwrite is not enabled.
    """

    _check_types(
        output_filetype=output_filetype,
        input_fps=input_fps,
        fps_from_filename=fps_from_filename,
        rotation=rotation,
        overwrite=overwrite,
        delete_input=delete_input,
    )

    output_fps = OUTPUT_FPS
    if output_fps is not None:
        delete_input = False  # Never delete input if re-encoding file.

    input_filename = input_video_file.stem
    input_filetype = input_video_file.suffix
    output_video_file = input_video_file.with_suffix(output_filetype)

    if not overwrite and output_video_file.is_file():
        log.warning(
            f"{output_video_file} already exists. To overwrite, set overwrite to True"
        )
        return None
    vid_filetypes = CONFIG["FILETYPES"]["VID"]

    if input_filetype in CONVERTABLE_FILETYPES and output_filetype in vid_filetypes:
        if fps_from_filename:
            input_fps = _get_fps_from_filename(input_filename)

        ffmpeg_cmd = _get_ffmpeg_command(
            input_video_file, input_fps, rotation, output_fps, output_video_file
        )

        subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        log.info(f"{output_video_file} created an input fps of {input_fps}")

        if delete_input:
            _delete_input_video_file(input_video_file, output_video_file)
    else:
        raise TypeError(f"Output video filetype {output_filetype} is not supported")


def _get_ffmpeg_command(
    input_video_file: Path,
    input_fps: float,
    rotation: int,
    output_fps: Optional[float],
    output_video_file: Path,
    filter_cmds: Optional[list[str]] = None,
) -> list[str]:
    """
    Generate an ffmpeg command using the given options.

    Args:
        input_video_file (Path): Path to h264 video file (or other format).
        input_fps (float, optional): Frame rate of input h264.
            If fps_from_filename is set to True, input_fps will be ignored.
        rotation (int, optional): Add rotation information to video metadata.
        output_fps (Optional[float]): Frame rate of the output file.
        output_video_file (Path): Path to the output video file.
        filter_cmds (Optional[list[str]]): Filter to use with ffmpeg. Filters (maybe
            necessary for special cases, insert if needed)

    Returns:

    """
    # ? Change -framerate to -r?
    input_fps_cmds = ["-r", str(input_fps)]

    if rotation == 0:
        rotation_cmds: list[str] = []
    else:
        if output_video_file.suffix not in CONFIG[FILETYPES][VID_ROTATABLE]:
            raise TypeError(
                f"{output_video_file.suffix} files are not rotatable."
                f"Use {CONFIG[FILETYPES][VID_ROTATABLE]} or rotation=0 instead."
            )
        rotation_cmds = ["-display_rotation", str(rotation)]

    if output_fps is not None:
        output_fps_cmds: list[str] = ["-r", str(output_fps)]
        copy_cmds: list[str] = []
    else:
        output_fps_cmds = []
        copy_cmds = ["-vcodec", "copy"]  # No re-encoding, only demuxing

    input_file_cmds = ["-i", str(input_video_file)]

    filter_cmds = filter_cmds if filter_cmds else []

    output_file_cmds = ["-y", str(output_video_file)]

    ffmpeg_cmd = (
        ["ffmpeg"]
        + rotation_cmds
        + input_fps_cmds
        + input_file_cmds
        + filter_cmds
        + output_fps_cmds
        + copy_cmds
        + output_file_cmds
    )
    log.debug(f"ffmpeg command: {ffmpeg_cmd}")
    return ffmpeg_cmd


def _delete_input_video_file(input_video_file: Path, output_video_file: Path) -> None:
    in_size = input_video_file.stat().st_size
    out_size = output_video_file.stat().st_size
    if in_size <= out_size:
        log.debug(f"Input file ({in_size}) <= output file ({out_size}).")
        input_video_file.unlink()
        log.info(f"Input file {input_video_file} deleted.")


def check_ffmpeg() -> None:
    """Checks, if ffmpeg is available"""

    exception_msg = "ffmpeg can not be called, check it's installed correctly"

    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        log.info("ffmpeg was found")
    except FileNotFoundError:
        log.exception(exception_msg)
        raise
    except subprocess.CalledProcessError:
        log.exception(exception_msg)
        raise
    except Exception:
        log.exception("")
        raise


def _check_types(
    output_filetype: str,
    input_fps: float,
    fps_from_filename: bool,
    rotation: int,
    overwrite: bool,
    delete_input: bool,
) -> None:
    """Raise ValueErrors if wrong types"""

    if not isinstance(output_filetype, str):
        raise ValueError("output_filetype has to be str")
    if not isinstance(input_fps, (int, float)):
        raise ValueError("input_fps has to be int or float")
    if not isinstance(fps_from_filename, bool):
        raise ValueError("fps_from_filename has to be bool")
    if not isinstance(rotation, int):
        raise ValueError("rotation has to be int")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite has to be bool")
    if not isinstance(delete_input, bool):
        raise ValueError("delete_input has to be bool")
