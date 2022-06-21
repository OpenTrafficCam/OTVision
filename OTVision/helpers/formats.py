import contextlib
import re
from typing import Union

import numpy as np
import pandas as pd


def _get_fps_from_filename(filename: str) -> int:
    """Get frame rate from file name using regex.

    Args:
        input_filename (str): file name

    Returns:
        int: frame rate in frames per second
    """
    with contextlib.suppress(AttributeError("Frame rate not found in filename")):
        # Get input fps frome filename  #TODO: Check regex for numbers
        input_fps = float(re.search("_FR(.*?)_", filename)[1])
    return input_fps


def _get_datetime_from_filename(
    filename: str, epoch_datetime="1970-01-01_00-00-00"
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
    try:
        # Get input fps frome filename  #TODO: Check regex for numbers
        yyyy = "[2]+[0-1]+[0-9]+[0-9]"
        mm = "[0-1]+[0-9]"
        dd = "[0-3]+[0-9]"
        hh = "[0-2]+[0-9]"
        mm = "[0-5]+[0-9]"
        ss = "[0-5]+[0-9]"
        expr = f"[_]+{yyyy}+[-]+{mm}+[-]+{dd}+[_]+{hh}+[-]+{mm}+[-]+{ss}"
        datetime_str = re.search(expr, filename)[0][1:]
    except AttributeError("Frame rate not found in filename"):
        datetime_str = epoch_datetime
    return datetime_str


def _ottrk_dict_to_df(nested_dict: dict) -> pd.DataFrame:
    """Turns a dict of tracks into a dataframe

    Args:
        nested_dict (dict): Nested dict of tracks from .ottrk file

    Returns:
        pd.DataFrame: DataFrame of tracks
    """
    return (
        pd.DataFrame.from_dict(
            {
                (i, j): nested_dict[i][j]
                for i in nested_dict
                for j in nested_dict[i].keys()
            },
            orient="index",
        )
        .reset_index()
        .rename(columns={"level_0": "frame", "level_1": "object"})
    )


def _get_time_from_frame_number(
    frame_series: pd.Series,
    start_datetime: str,
    fps: int,
    return_yyyymmdd_hhmmss=True,
    return_milliseconds=True,
) -> Union[pd.Series, pd.Series]:
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
    elif not return_yyyymmdd_hhmmss:
        return datetime_milliseconds
    else:
        raise ValueError(
            "Either return_yyyymmdd_hhmmss or return_milliseconds has to be True"
        )


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
