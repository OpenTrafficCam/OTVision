"""
OTVision helpers to change formats and retrieve information
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


import datetime as dt
import re

import numpy as np
import pandas as pd


def _get_fps_from_filename(filename: str) -> int:
    """Get frame rate from file name using regex.
    Returns None if frame rate is not found in file name.

    Args:
        input_filename (str): file name

    Returns:
        int or None: frame rate in frames per second or None
    """

    if match := re.search(r"_FR([\d]+)_", filename):
        return int(match[1])
    else:
        raise ValueError(f"Cannot read frame rate from file name {filename}")


def _get_datetime_from_filename(
    filename: str, epoch_datetime: str = "1970-01-01_00-00-00"
) -> str:
    """Get date and time from file name.
    Searches for "_yyyy-mm-dd_hh-mm-ss".
    Returns "yyyy-mm-dd_hh-mm-ss".

    Args:
        filename (str): filename with expression
        epoch_datetime (str): Unix epoch (00:00:00 on 1 January 1970)

    Returns:
        str: datetime
    """
    regex = "_([0-9]{4,4}-[0-9]{2,2}-[0-9]{2,2}_[0-9]{2,2}-[0-9]{2,2}-[0-9]{2,2})"
    match = re.search(regex, filename)
    if not match:
        return epoch_datetime

    # Assume that there is only one timestamp in the file name
    datetime_str = match[1]

    try:
        dt.datetime.strptime(datetime_str, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return epoch_datetime

    return datetime_str


# TODO: Type hint nested dict during refactoring
def _ottrk_detections_to_df(ottrk: list) -> pd.DataFrame:
    """Turns a list of tracked detections into a dataframe

    Args:
        ottrk (list): List of dicts, each dict is a tracked detection.
            Comes from from .ottrk file.

    Returns:
        pd.DataFrame: DataFrame of tracks
    """
    return pd.DataFrame(ottrk)


def _get_time_from_frame_number(
    frame_series: pd.Series,
    start_datetime: str,
    fps: int,
    return_yyyymmdd_hhmmss: bool = True,
    return_milliseconds: bool = True,
) -> pd.Series | tuple[pd.Series, pd.Series]:
    """Get datetime series of detections from series of frame numbers of video
    the objects were detected using a start datetime of the video and
    the video frame rate (fps).

    Args:
        frame_series (pd.Series): Series of video frames of detections
        start_datetime (str): Start datetime of video
        fps (int): Video frame rate in frames per second

    Returns:
        pd.Series: Datetime series of detections in "%Y-%m-%d_%H-%M-%S" format
        pd.Series: Datetime series of detections in milliseconds
    """
    datetime_yyyymmdd_hhmmss = pd.to_datetime(
        start_datetime, format=r"%Y-%m-%d_%H-%M-%S"
    ) + pd.to_timedelta((frame_series.astype("int32") - 1) / fps, unit="s")
    if return_milliseconds:
        datetime_milliseconds = datetime_yyyymmdd_hhmmss.astype(np.int64) / int(1e6)
    if return_yyyymmdd_hhmmss and return_milliseconds:
        return datetime_yyyymmdd_hhmmss, datetime_milliseconds
    elif not return_milliseconds:
        return datetime_yyyymmdd_hhmmss
    else:
        return datetime_milliseconds


def _get_epsg_from_utm_zone(utm_zone: int, hemisphere: str) -> int:
    """Calculates the epsg number from utm zone and hemisphere.

    Args:
        utm_zone (int): UTM zone (1-60)
        hemisphere (str): Hemisphere ("N" or "S")

    Returns:
        int: epsg number of UTM zone (e.g. 32632)
    """
    identifier_digits = 32000
    if hemisphere == "N":
        hemisphere_digit = 600
    elif hemisphere == "S":
        hemisphere_digit = 700
    return identifier_digits + hemisphere_digit + utm_zone
