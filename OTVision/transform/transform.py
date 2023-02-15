"""
OTVision main module for transforming tracks from pixel to world coordinates
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


from copy import deepcopy
from pathlib import Path
from typing import Union

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd

from OTVision.config import CONFIG, FILETYPES, REFPTS, TRACK
from OTVision.helpers.files import (
    _check_and_update_metadata_inplace,
    get_files,
    read_json,
    replace_filetype,
    write_json,
)
from OTVision.helpers.formats import (
    _get_datetime_from_filename,
    _get_epsg_from_utm_zone,
    _get_time_from_frame_number,
    _ottrk_dict_to_df,
)
from OTVision.helpers.log import log, reset_debug, set_debug

from .get_homography import get_homography


def main(
    paths: list[Path],
    refpts_file: Union[Path, None] = None,
    overwrite: bool = CONFIG["TRANSFORM"]["OVERWRITE"],
    debug: bool = CONFIG["TRANSFORM"]["DEBUG"],
) -> None:
    """Transform tracks files containing trajectories in pixel coordinates to .gpkg
    files with trajectories in utm coordinates using either one single refpts file for
    all tracks files containing corresponding reference points in both pixel and utm
    coordinates or using specific refpts files for each tracks file
    (path must be the same except for the extension).
    Info: refpts file can be created using OTVision.transform.reference_points_picker

    Args:
        paths (list[Path]): List of paths to tracks files.
        refpts_file (Path, optional): Path to reference points file.
            If given, this file will be used for transformation of all tracks files.
            If not given, for every tracks file a refpts file with same stem is needed.
            Defaults to None.
        overwrite (bool, optional): Whether or not to overwrite existing tracks files in
            world coordinates.
            Defaults to CONFIG["TRANSFORM"]["OVERWRITE"].
        debug (bool, optional): Whether or not to run in debug mode.
            Defaults to CONFIG["TRANSFORM"]["DEBUG"].
    """

    log.info("Start transformation from pixel to world coordinates")
    if debug:
        set_debug()

    track_filetype = CONFIG[FILETYPES][TRACK]
    refpts_filetype = CONFIG[FILETYPES][REFPTS]

    if refpts_file:
        if not refpts_file.exists():
            raise FileNotFoundError(
                f"No reference points file with filetype: '{refpts_filetype}' found!"
            )

        refpts = read_refpts(reftpts_file=refpts_file)
        (
            homography,
            refpts_utm_upshift_predecimal,
            upshift_utm,
            utm_zone,
            hemisphere,
            homography_eval_dict,
        ) = get_homography(refpts=refpts)
        log.info(f"Read {refpts_file}")
    tracks_files = get_files(paths=paths, filetypes=CONFIG["FILETYPES"]["TRACK"])
    if not tracks_files:
        raise FileNotFoundError(
            f"No files of type '{track_filetype}' found to transform!"
        )

    for tracks_file in tracks_files:
        log.info(f"Try transforming {tracks_file}")
        # Try reading refpts and getting homography if not done above
        if not refpts_file:
            associated_refpts_file = replace_filetype(
                files=[tracks_file], new_filetype=CONFIG["DEFAULT_FILETYPE"]["REFPTS"]
            )[0]
            if not associated_refpts_file.exists():
                raise FileNotFoundError(
                    f"No reference points file found for tracks file: {tracks_file}!"
                )

            log.info(f"Found refpts file {associated_refpts_file}")
            refpts = read_refpts(reftpts_file=associated_refpts_file)
            log.info("Refpts read")
            (
                homography,
                refpts_utm_upshift_predecimal,
                upshift_utm,
                utm_zone,
                hemisphere,
                homography_eval_dict,
            ) = get_homography(refpts=refpts)
            log.info("Homography matrix created")
        # Read tracks
        tracks_px_df, metadata_dict = read_tracks(tracks_file)
        log.info("Tracks read")
        # Check if transformation is actually needed
        # already_utm = "utm" in metadata_dict["trk"] and metadata_dict["trk"]["utm"]
        if overwrite:  # ? or not already_utm?
            metadata_dict["trk"]["utm"] = False  # ? or not?
            # Actual transformation
            tracks_utm_df = transform(
                tracks_px=tracks_px_df,
                homography=homography,
                refpts_utm_upshifted_predecimal_pt1_1row=refpts_utm_upshift_predecimal,
                upshift_utm=upshift_utm,
            )
            log.info("Tracks transformed")
            # Add crs information tp metadata dict
            metadata_dict["trk"]["utm"] = True
            metadata_dict["trk"]["utm_zone"] = utm_zone
            metadata_dict["trk"]["hemisphere"] = hemisphere
            metadata_dict["trk"]["epsg"] = _get_epsg_from_utm_zone(
                utm_zone=utm_zone, hemisphere=hemisphere
            )
            metadata_dict["trk"]["transformation accuracy"] = homography_eval_dict
            log.info("Meta infos created")
            log.debug(f"config_dict: {metadata_dict}")
            # Write tracks
            write_tracks(
                tracks_utm_df=tracks_utm_df,
                metadata_dict=metadata_dict,
                utm_zone=utm_zone,
                hemisphere=hemisphere,
                tracks_file=tracks_file,
            )
        log.info("Transformation successful")
    if debug:
        reset_debug()


# TODO: Type hint nested dict during refactoring
def read_tracks(tracks_file: Path) -> tuple[pd.DataFrame, dict]:
    """Reads .ottrk file, returns pandas DataFrame of trajectories
    in pixel coordinates and dict of metadata

    Args:
        tracks_file (str or Pathlib.Path): Path to .ottrk file

    Returns:
        pandas.DataFrame: DataFrame of trajectories in pixel coordinates
        dict: Dict of metadata
    """

    # Read dicts and turn tracks into DataFrame
    tracks_dict = read_json(tracks_file, filetype=tracks_file.suffix)
    _check_and_update_metadata_inplace(tracks_dict)
    tracks_df = _ottrk_dict_to_df(tracks_dict["data"])
    metadata_dict = tracks_dict["metadata"]

    # Create datetime column from frame number
    fps = int(metadata_dict["vid"]["fps"])
    start_datetime = _get_datetime_from_filename(filename=str(tracks_file))
    tracks_df["datetime"], tracks_df["datetime_ms"] = _get_time_from_frame_number(
        frame_series=tracks_df["frame"], start_datetime=start_datetime, fps=fps
    )
    return tracks_df, metadata_dict


def read_refpts(
    reftpts_file: Path,
) -> dict:  # TODO: Type hint nested dict during refactoring
    """Reads .otrfpts file, returns dict of matching reference points
    in both pixel and utm coordinates

    Args:
        reftpts_file (str or pathlib.Path): Path to .rfpts file

    Returns:
        dict: Matching reference points in both pixel and utm coordinates
    """

    return read_json(reftpts_file, filetype=reftpts_file.suffix, decompress=False)


def transform(
    tracks_px: pd.DataFrame,
    homography: np.ndarray,
    refpts_utm_upshifted_predecimal_pt1_1row: np.ndarray,
    upshift_utm: np.ndarray,
    x_col: str = "x",
    y_col: str = "y",
    lon_col: str = "lon_utm",
    lat_col: str = "lat_utm",
) -> pd.DataFrame:
    """Transforms trajectories from pixel to utm coordinates using homography from
    get_homography using corresponding reference points, adds utm coordinates to
    trajectories

    Args:
        tracks_px (pandas.DataFrame): Trajectories in pixel coordinates
        homography (numpy.ndarry): Homography matrix between pixel and utm coordinates
        refpts_utm_upshifted_predecimal_pt1_1row (numpy.ndarry): Thousands digits
            of reference points in utm coorindates
        upshift_utm (numpy.ndarry): Upshift of reference points coordinates
        x_col (str, optional): Column name of x-pixel values. Defaults to "x".
        y_col (str, optional): Column name of y-pixel values. Defaults to "y".
        lon_col (str, optional): Column name of lon values. Defaults to "lon_utm".
        lat_col (str, optional): Column name of lat values. Defaults to "lat_utm".

    Returns:
        pandas.DataFrame: Trajectories in utm coordinates
    """

    tracks_utm = deepcopy(tracks_px)

    # Transform pandas DataFrame to numpy array, add 1 dimension and apply OpenCV´s
    # perspective transformation
    tracks_px_np = tracks_px[[x_col, y_col]].to_numpy(dtype="float32")
    tracks_px_np_tmp = np.array([tracks_px_np], dtype="float32")
    tracks_utm_upshifted_np_disassembled_3d = cv2.perspectiveTransform(
        tracks_px_np_tmp, homography
    )
    tracks_utm_upshifted_np_disassembled = np.squeeze(
        tracks_utm_upshifted_np_disassembled_3d
    )

    # Concatenate the thousands digits truncated before transformation
    tracks_utm_upshifted_np = np.add(
        np.multiply(refpts_utm_upshifted_predecimal_pt1_1row, 1000),
        tracks_utm_upshifted_np_disassembled,
    )

    # Shift back down both y and y world coordinates (same values as reference points
    # were upshifted)
    tracks_utm_np = np.subtract(tracks_utm_upshifted_np, upshift_utm)

    # In trajectory DataFrame, overwrite pixel with utm coordinates (from numpy array)
    tracks_utm[[lon_col, lat_col]] = tracks_utm_np

    return tracks_utm


# TODO: Type hint nested dict during refactoring
def write_refpts(refpts: dict, refpts_file: Path) -> None:
    """Writes corresponding reference points in both pixel and utm coordinates
    to a json-like .otrfpts file

    Args:
        refpts (dict): Corresponding reference points in both pixel and utm coordinates
        refpts_file (str or pathlib.Path): Path of the refpts file to be written
    """
    write_json(
        dict_to_write=refpts,
        file=refpts_file,
        filetype=CONFIG["DEFAULT_FILETYPE"]["REFPTS"],
        overwrite=True,
    )


# TODO: Type hint nested dict during refactoring
def write_tracks(
    tracks_utm_df: pd.DataFrame,
    metadata_dict: dict,
    utm_zone: int,
    hemisphere: str,
    tracks_file: Path,
    filetype: str = "gpkg",
    overwrite: bool = CONFIG["TRANSFORM"]["OVERWRITE"],
) -> None:
    """Writes tracks as .gpkg and in specific epsg projection
    according to UTM zone and hemisphere

    Args:
        tracks_utm_df (geopandas.GeoDataFrame): Trajectories in utm coordinates
        config_dict (dict): Meta data dict
        utm_zone (str): UTM zone.
        hemisphere (str): Hemisphere where trajectories were recorded. "N" or "S".
        tracks_file (str or pathlib.Path): Path to tracks file (in pixel coordinates)
        filetype (str, optional): _description_. Defaults to "gpkg".
    """

    # TODO: Write metadata

    if filetype == "gpkg":  # TODO: Extend guard with overwrite parameter
        gpkg_file = tracks_file.with_suffix(".gpkg")
        gpkg_file_already_exists = gpkg_file.is_file()
        if overwrite or not gpkg_file_already_exists:
            # Get CRS UTM EPSG number
            epsg = _get_epsg_from_utm_zone(utm_zone=utm_zone, hemisphere=hemisphere)
            # Create, set crs and write geopandas.GeoDataFrame
            gpd.GeoDataFrame(
                tracks_utm_df,
                geometry=gpd.points_from_xy(
                    tracks_utm_df["lon_utm"], tracks_utm_df["lat_utm"]
                ),
            ).set_crs(f"epsg:{epsg}").to_file(filename=gpkg_file, driver="GPKG")
            if gpkg_file_already_exists:
                log.info(f"{gpkg_file} overwritten")
            else:
                log.info(f"{gpkg_file}  file written")
        else:
            log.info(f"{gpkg_file} already exists. To overwrite, set overwrite=True")
    # TODO: Export tracks as ottrk (json)
