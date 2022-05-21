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


import json
from copy import deepcopy
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files, read_json, write_json
from OTVision.helpers.formats import get_epsg_from_utm_zone, ottrk_dict_to_df
from OTVision.transform.get_homography import get_homography


def main(tracks_files, reftpts_file):
    # ? Separate homography for each tracks file, as each has refpts file? How expensive?
    reftpts = read_refpts(reftpts_file=reftpts_file)
    (
        homography,
        refpts_utm_upshifted_predecimal_pt1_1row,
        upshift_utm,
        utm_zone,
        hemisphere,
    ) = get_homography(refpts=reftpts)
    tracks_files = get_files(paths=tracks_files, filetypes=CONFIG["FILETYPES"]["TRACK"])
    for tracks_file in tracks_files:
        tracks_px_df, config_dict = read_tracks(tracks_file)
        already_utm = (
            "utm" in config_dict["trk_config"] and config_dict["trk_config"]["utm"]
        )
        if CONFIG["TRANSFORM"]["OVERWRITE"] or not already_utm:
            config_dict["trk_config"]["utm"] = False  # or not?
            # Actual transformation
            tracks_utm_df = transform(
                tracks_px=tracks_px_df,
                homography=homography,
                refpts_utm_upshifted_predecimal_pt1_1row=refpts_utm_upshifted_predecimal_pt1_1row,
                upshift_utm=upshift_utm,
            )
            # Add crs information tp config dict
            config_dict["trk_config"]["utm"] = True
            config_dict["trk_config"]["utm_zone"] = utm_zone
            config_dict["trk_config"]["hemisphere"] = hemisphere
            config_dict["trk_config"]["epsg"] = get_epsg_from_utm_zone(
                utm_zone=utm_zone, hemisphere=hemisphere
            )
            # Write tracks and
            write_tracks(
                tracks_utm_df=tracks_utm_df,
                config_dict=config_dict,
                utm_zone=utm_zone,
                hemisphere=hemisphere,
                tracks_file=tracks_file,
            )


def read_tracks(tracks_file):
    tracks_dict = read_json(tracks_file, extension=CONFIG["FILETYPES"]["TRACK"])
    tracks_df = ottrk_dict_to_df(tracks_dict["data"])
    config_dict = {key: value for key, value in tracks_dict.items() if key != "data"}
    return tracks_df, config_dict


def read_refpts(reftpts_file):
    return read_json(reftpts_file, extension=CONFIG["FILETYPES"]["REFPTS"])


def transform(
    tracks_px,
    homography,
    refpts_utm_upshifted_predecimal_pt1_1row,
    upshift_utm,
):
    """
    Convert trajectories from pixel to world coordinates using homography matrix.
    Keyword arguments:
    traj_pixel -- Trajectory points in pixel coordinates
    homography -- Homography matrix gathered from reference points in pixel and
    world coordinates
    refpts_world_upshifted_predecimal_pt1_1row -- Thousands digits of reference points
    in world coorindates
    upshift_world -- Formerly performed upshift of reference points coordinates
    """

    tracks_utm = deepcopy(tracks_px)

    # Transform pandas dataframe to numpy array, add 1 dimension and apply OpenCV´s
    # perspective transformation
    tracks_px_np = tracks_px[["x", "y"]].to_numpy(
        dtype="float32"
    )  # TODO: Rename to "x_py" and "y_px"
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

    # In trajectory dataframe, overwrite pixel coordinates with world coordinates
    # (from numpy array)
    tracks_utm = tracks_px
    tracks_utm[["lon_utm", "lat_utm"]] = tracks_utm_np

    return tracks_utm


def write_refpts(refpts, refpts_file):
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

    # TODO: Write config dict

    if filetype == "gpkg":
        outfile = Path(tracks_file).with_suffix(".gpkg")
        if not outfile.is_file() or CONFIG["TRANSFORM"]["OVERWRITE"]:
            # Get CRS UTM EPSG number
            epsg = get_epsg_from_utm_zone(utm_zone=utm_zone, hemisphere=hemisphere)
            # Create and write geodataframe
            gdf_tracks_utm = (
                gpd.GeoDataFrame(
                    tracks_utm_df,
                    geometry=gpd.points_from_xy(
                        tracks_utm_df["lon_utm"], tracks_utm_df["lat_utm"]
                    ),
                )
                .set_crs(f"epsg:{epsg}")
                .to_file(filename=outfile, driver="GPKG")
            )
    # TODO: Export tracks as ottrk (json)
    # elif filetype == CONFIG["DEFAULT_FILETYPE"]["TRACKS"]:
    #     write_json(
    #         dict_to_write=tracks_utm,
    #         file=tracks_file,
    #         extension=".ottrk",
    #         overwrite=True,
    #     )
