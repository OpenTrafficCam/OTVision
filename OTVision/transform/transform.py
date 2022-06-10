# OTVision: Python module to transform tracks from pixel to world coordinates

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

import cv2
import geopandas as gpd
import numpy as np

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files, read_json, write_json
from OTVision.helpers.formats import get_epsg_from_utm_zone, ottrk_dict_to_df
from OTVision.transform.get_homography import get_homography


def main(tracks_files, single_refpts_file=None):
    """Transforms .ottrk file containing trajectories in pixel coordinates
    to .gpkg file with trajectories in utm coordinates
    using either a single .otrfpts file for all .ottrk files
    containing corresponding reference points in both pixel and utm coordinates
    or using specific .otrfpts files for each .ottrk file
    (path must be the same except for the extension).

    Info: .otrfpts file can be created using OTVision.transform.reference_points_picker

    Args:
        tracks_files (str or Path or list): (List of) Path(s) to .ottrk file
        single_refpts_file (str or Path, optional): Path to .otrfpts file. Default: None
    """

    if single_refpts_file:
        refpts = read_refpts(reftpts_file=single_refpts_file)
        (
            homography,
            refpts_utm_upshift_predecimal,
            upshift_utm,
            utm_zone,
            hemisphere,
            homography_eval_dict,
        ) = get_homography(refpts=refpts)
    tracks_files = get_files(paths=tracks_files, filetypes=CONFIG["FILETYPES"]["TRACK"])
    for tracks_file in tracks_files:
        # Try reading refpts and getting homography if not done above
        if not single_refpts_file:
            refpts_file = get_files(
                paths=tracks_file,
                filetypes=CONFIG["DEFAULT_FILETYPE"]["REFPTS"],
                replace_filetype=True,
            )[0]
            refpts = read_refpts(reftpts_file=refpts_file)
            (
                homography,
                refpts_utm_upshift_predecimal,
                upshift_utm,
                utm_zone,
                hemisphere,
                homography_eval_dict,
            ) = get_homography(refpts=refpts)
        # Read tracks
        tracks_px_df, config_dict = read_tracks(tracks_file)
        # Check if transformation is actually needed
        already_utm = (
            "utm" in config_dict["trk_config"] and config_dict["trk_config"]["utm"]
        )
        if CONFIG["TRANSFORM"]["OVERWRITE"] or not already_utm:
            config_dict["trk_config"]["utm"] = False  # ? or not?
            # Actual transformation
            tracks_utm_df = transform(
                tracks_px=tracks_px_df,
                homography=homography,
                refpts_utm_upshifted_predecimal_pt1_1row=refpts_utm_upshift_predecimal,
                upshift_utm=upshift_utm,
            )
            # Add crs information tp config dict
            config_dict["trk_config"]["utm"] = True
            config_dict["trk_config"]["utm_zone"] = utm_zone
            config_dict["trk_config"]["hemisphere"] = hemisphere
            config_dict["trk_config"]["epsg"] = get_epsg_from_utm_zone(
                utm_zone=utm_zone, hemisphere=hemisphere
            )
            config_dict["trk_config"]["transformation accuracy"] = homography_eval_dict
            print(config_dict)
            # Write tracks
            write_tracks(
                tracks_utm_df=tracks_utm_df,
                config_dict=config_dict,
                utm_zone=utm_zone,
                hemisphere=hemisphere,
                tracks_file=tracks_file,
            )


def read_tracks(tracks_file):
    """Reads .ottrk file, returns pandas DataFrame of trajectories
    in pixel coordinates and dict of metadata

    Args:
        tracks_file (str or Pathlib.Path): Path to .ottrk file

    Returns:
        pandas.DataFrame: DataFrame of trajectories in pixel coordinates
        dict: Dict of metadata
    """

    tracks_dict = read_json(tracks_file, extension=CONFIG["FILETYPES"]["TRACK"])
    tracks_df = ottrk_dict_to_df(tracks_dict["data"])
    config_dict = {key: value for key, value in tracks_dict.items() if key != "data"}
    return tracks_df, config_dict


def read_refpts(reftpts_file):
    """Reads .otrfpts file, returns dict of matching reference points
    in both pixel and utm coordinates

    Args:
        reftpts_file (str or pathlib.Path): Path to .rfpts file

    Returns:
        dict: Matching reference points in both pixel and utm coordinates
    """

    return read_json(reftpts_file, extension=CONFIG["FILETYPES"]["REFPTS"])


def transform(
    tracks_px,
    homography,
    refpts_utm_upshifted_predecimal_pt1_1row,
    upshift_utm,
):
    """Transforms trajectories from pixel to utm coordinates using homography from
    get_homography using corresponding reference points,
    adds utm coordinates to trajectories

    Args:
        tracks_px (pandas.DataFrame): Trajectories in pixel coordinates
        homography (numpy.ndarry): Homography matrix between pixel and utm coordinates
        refpts_utm_upshifted_predecimal_pt1_1row (numpy.ndarry): Thousands digits
        of reference points in utm coorindates
        upshift_utm (numpy.ndarry): Upshift of reference points coordinates

    Returns:
        pandas.DataFrame: Trajectories in utm coordinates
    """

    tracks_utm = deepcopy(tracks_px)

    # Transform pandas DataFrame to numpy array, add 1 dimension and apply OpenCV´s
    # perspective transformation
    tracks_px_np = tracks_px[["x", "y"]].to_numpy(dtype="float32")
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
    tracks_utm[["lon_utm", "lat_utm"]] = tracks_utm_np

    return tracks_utm


def write_refpts(refpts, refpts_file):
    """Writes corresponding reference points in both pixel and utm coordinates
    to a json-like .otrfpts file

    Args:
        refpts (dict): Corresponding reference points in both pixel and utm coordinates
        refpts_file (str or pathlib.Path): Path of the refpts file to be written
    """
    write_json(
        dict_to_write=refpts,
        file=refpts_file,
        extension=CONFIG["DEFAULT_FILETYPE"]["REFPTS"],
        overwrite=True,
    )


def write_tracks(
    tracks_utm_df,
    config_dict,
    utm_zone,
    hemisphere,
    tracks_file,
    filetype="gpkg",
):
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

    # TODO: Write config dict

    if filetype == "gpkg":  # TODO: Extend guard with overwrite parameter
        outfile = Path(tracks_file).with_suffix(".gpkg")
        if not outfile.is_file() or CONFIG["TRANSFORM"]["OVERWRITE"]:
            # Get CRS UTM EPSG number
            epsg = get_epsg_from_utm_zone(utm_zone=utm_zone, hemisphere=hemisphere)
            # Create and write geopandas.GeoDataFrame
            gpd.GeoDataFrame(
                tracks_utm_df,
                geometry=gpd.points_from_xy(
                    tracks_utm_df["lon_utm"], tracks_utm_df["lat_utm"]
                ),
            ).rename(columns={"level_0": "frame", "level_1": "object"}).set_crs(
                f"epsg:{epsg}"
            ).to_file(
                filename=outfile, driver="GPKG"
            )
    # BUG: Rename doesnt work :/
    # ? remove lat lon cols from gdf after creating geom col?
    # TODO: Export tracks as ottrk (json)
    # elif filetype == CONFIG["DEFAULT_FILETYPE"]["TRACKS"]:
    #     write_json(
    #         dict_to_write=tracks_utm,
    #         file=tracks_file,
    #         extension=".ottrk",
    #         overwrite=True,
    #     )
