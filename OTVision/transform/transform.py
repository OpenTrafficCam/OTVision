# OTVision: Python module to calculate homography matrix from reference
# points and transform trajectory points from pixel into world coordinates.

# Copyright (C) 2020 OpenTrafficCam Contributors
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

"""
This module calculates a homography matrix and transforms trajectory points from pixel
into world coordinates.
It requires reference points in both pixel and world coordinates as well as
trajectories in pixel coordinates.
Reference points have to be loaded from .npy files (or from semicolon delimited .txt
files without headers or indices).
Multiple trajectory files with data arranged in otc style (OpenTrafficCam) can be
selected,  preferably from .pkl files.
The module saves the trajectories in world coordinates as .pkl files.
For different camera views, the module has to be run repeatedly with respective
reference points in pixel coordinates.

To Dos:
- Save also homography as npy file?
"""

# TODO: #104 OpenCV findHomography and perspectiveTransform cannot handle after delimiter digits

import os
import logging
import cv2
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from helpers.files import get_files, read_json
import transform.helpers as helpers
from pathlib import Path


# Define relative path to test data (using os.path.dirname repeatedly)
TEST_DATA_FOLDER = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    + r"\tests\data"
)


def main(tracks_files, refpts_file):
    refpts = pd.read_csv(refpts_file, delimiter=";")
    (
        homography,
        refpts_world_upshifted_predecimal_pt1_1row,
        upshift_world,
    ) = get_homography(refpts)
    tracks_files = get_files(paths=tracks_files, filetypes=["GeoJSON", "gpkg"])
    homography = get_homography(refpts)
    for tracks_file in tracks_files:
        tracks_px = gpd.read_file(tracks_file)
        tracks_utm = transform_to_utm(
            tracks_px,
            homography,
            refpts_world_upshifted_predecimal_pt1_1row,
            upshift_world,
        )
        outfile = Path(tracks_file).with_stem(Path(tracks_file).stem + "_utm")
        fiona_drivers_dict = {".gpkg": "GPKG", ".geojson": "GeoJSON"}
        tracks_utm.to_file(
            outfile,
            driver=fiona_drivers_dict[outfile.suffix],
        )
    # Add geojson + gpkg


def get_homography(refpts):
    """Calculate homography matrix using pixel and world coordinates of corresponding
    reference points.

    Keyword arguments:
    refpts -- csv file of reference points in both pixel and world coordinates ("X;Y;Lat;Lon")
    """

    # Transform pandas dataframe to numpy array, add 1 dimension and apply OpenCV´s
    # perspective transformation
    refpts_world = refpts[["Lat", "Lon"]].to_numpy(dtype="float32")
    refpts_world = np.array(refpts_world, dtype="float32")
    refpts_pixel = refpts[["X", "Y"]].to_numpy(dtype="float32")
    refpts_pixel = np.array(refpts_pixel, dtype="float32")

    # Upshift both x and y world coordinates of reference points to next round 500m
    # value (UTM is in meters)
    min = np.amin(refpts_world, axis=0)
    max = np.amax(refpts_world, axis=0)
    mean = np.divide(np.add(min, max), 2)
    mean_predecimal = mean.astype(int)
    mean_predecimal_pt1 = np.divide(mean_predecimal, 1000).astype(int)
    mean_predecimal_pt1_Plus_500 = np.add(mean_predecimal_pt1.astype(float), 0.5)
    mean_Plus_500 = np.multiply(mean_predecimal_pt1_Plus_500, 1000)
    upshift_world = np.subtract(mean_Plus_500, mean)
    refpts_world_upshifted = np.add(refpts_world, upshift_world)

    # Truncate thousands digits from shifted reference points
    refpts_world_upshifted_postdecimal = np.mod(refpts_world_upshifted, 1)
    refpts_world_upshifted_predecimal = refpts_world_upshifted.astype(int)
    refpts_world_upshifted_predecimal_pt1 = np.divide(
        refpts_world_upshifted_predecimal, 1000
    ).astype(int)
    refpts_world_upshifted_predecimal_pt1_1row = np.array(
        [
            [
                refpts_world_upshifted_predecimal_pt1.item(0),
                refpts_world_upshifted_predecimal_pt1.item(1),
            ]
        ]
    )
    refpts_world_upshifted_predecimal_pt2 = np.mod(
        refpts_world_upshifted_predecimal, 1000
    )
    refpts_world_upshifted_disassembled = np.add(
        refpts_world_upshifted_predecimal_pt2, refpts_world_upshifted_postdecimal
    )

    # Calculate homography matrix with refpts in pixel coordinates and truncated &
    # shifted refpts in world coordinates
    homography, mask = cv2.findHomography(
        refpts_pixel, refpts_world_upshifted_disassembled, cv2.RANSAC, 3.0
    )  # RANSAC: Otulier/Inlier definieren??? # FEHLER:
    print(homography)
    print(mask)

    # Evaluate accuracy of homography matrix using reference points in world coords
    refpts_pixel_tmp = np.array([refpts_pixel], dtype="float32")
    refpts_world_upshifted_disassembled_transf_3d = cv2.perspectiveTransform(
        refpts_pixel_tmp, homography
    )
    refpts_world_upshifted_disassembled_transf = np.squeeze(
        refpts_world_upshifted_disassembled_transf_3d
    )
    eval_df = pd.DataFrame(
        {
            "x_ref": refpts_world_upshifted_disassembled[:, 0],
            "y_ref": refpts_world_upshifted_disassembled[:, 1],
            "x_transf": refpts_world_upshifted_disassembled_transf[:, 0],
            "y_transf": refpts_world_upshifted_disassembled_transf[:, 1],
        }
    )
    eval_df["x_delta"] = eval_df["x_transf"] - eval_df["x_ref"]
    eval_df["y_delta"] = eval_df["y_transf"] - eval_df["y_ref"]
    # Normalize error vectors using sentence of pythagoras
    eval_df["delta"] = np.linalg.norm(eval_df[["x_delta", "y_delta"]].values, axis=1)
    eval_df["delta_abs"] = eval_df["delta"].abs()
    print("Mean transformation error [m]: " + str(eval_df["delta_abs"].mean()))
    print("Maximum transformation error [m]: " + str(eval_df["delta_abs"].max()))

    return homography, refpts_world_upshifted_predecimal_pt1_1row, upshift_world


def transform_to_utm(
    tracks_px,
    homography,
    refpts_world_upshifted_predecimal_pt1_1row,
    upshift_world,
):
    """Convert trajectories from pixel to world coordinates using homography matrix.

    Keyword arguments:
    tracks_px -- Trajectory points in pixel coordinates
    homography -- Homography matrix gathered from reference points in pixel and
    world coordinates
    refpts_world_upshifted_predecimal_pt1_1row -- Thousands digits of reference points
    in world coorindates
    upshift_world -- Formerly performed upshift of reference points coordinates
    """

    # Transform pandas dataframe to numpy array, add 1 dimension and apply OpenCV´s
    # perspective transformation
    tracks_px_np = tracks_px[["x", "y"]].to_numpy(dtype="float32")
    tracks_px_np_tmp = np.array([tracks_px_np], dtype="float32")
    tracks_utm_upshifted_np_disassembled_3d = cv2.perspectiveTransform(
        tracks_px_np_tmp, homography[0]
    )
    tracks_utm_upshifted_np_disassembled = np.squeeze(
        tracks_utm_upshifted_np_disassembled_3d
    )

    # Concatenate the thousands digits truncated before transformation
    tracks_utm_upshifted_np = np.add(
        np.multiply(refpts_world_upshifted_predecimal_pt1_1row, 1000),
        tracks_utm_upshifted_np_disassembled,
    )

    # Shift back down both y and y world coordinates (same values as reference points
    # were upshifted)
    tracks_utm_np = np.subtract(tracks_utm_upshifted_np, upshift_world)

    # In trajectory dataframe, overwrite pixel coordinates with world coordinates
    # (from numpy array)
    tracks_utm = tracks_px
    # TODO: #105 resulting utm coordinates are still wrong
    tracks_utm[["lon", "lat"]] = tracks_utm_np

    # Overwrite geometry column and set crs
    tracks_utm["geometry"] = gpd.points_from_xy(tracks_utm["lat"], tracks_utm["lon"])
    tracks_utm = tracks_utm.set_crs(epsg=32633)

    return tracks_utm


def save_tracks_utm(tracks_px_path, tracks_utm):
    """Save trajectories in world coordinates as python pickle files and csv files.

    Keyword arguments:
    tracks_px_path -- Path of converted trajectories in pixel coordinates
    tracks_utm -- Trajectories in world coordinates
    """

    tracks_utm.to_csv(tracks_px_path[:-4] + "World_decDot.csv", index=False, sep=";")
    tracks_utm.to_csv(
        tracks_px_path[:-4] + "World_decComma.csv", index=False, sep=";", decimal=","
    )
    tracks_utm.to_pickle(tracks_px_path[:-4] + "World.pkl")


if __name__ == "__main__":
    refpts_pixel_path, refpts_world_path = helpers.select_refpts_files()
    tracks_px_paths = helpers.select_traj_files()
    main(refpts_pixel_path, refpts_world_path, tracks_px_paths)
