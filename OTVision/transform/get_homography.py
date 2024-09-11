"""
OTVision module for calculating a homography from reference points
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

import cv2
import numpy as np
import pandas as pd

from OTVision.helpers.log import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)


# TODO: Type hint nested dict during refactoring
def get_homography(
    refpts: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str, dict]:
    """Calculate homography matrix using pixel and world coordinates of corresponding
    reference points.

    Args:
        refpts (dict): Corresponding reference points in both pixel and utm coordinates

    Returns:
        ndarry: homography
        ndarry: refpts_utm_upshifted_predecimal_pt1_1row
        ndarry: upshift_utm
        int: utm_zone
        str: hemisphere
        dict: precision of homography
    """

    refpts_df = pd.DataFrame.from_dict(refpts, orient="index")
    refpts_px = refpts_df[["x_px", "y_px"]].to_numpy()
    refpts_utm = refpts_df[["lon_utm", "lat_utm"]].to_numpy()

    # Upshift both x and y world coordinates of reference points to next round 500m
    # value (UTM is in meters)
    refpts_utm_min = np.amin(refpts_utm, axis=0)
    refpts_utm_max = np.amax(refpts_utm, axis=0)
    refpts_utm_mean = np.divide(np.add(refpts_utm_min, refpts_utm_max), 2)
    mean_predecimal = refpts_utm_mean.astype(int)
    mean_predecimal_pt1 = np.divide(mean_predecimal, 1000).astype(int)
    mean_predecimal_pt1_Plus_500 = np.add(mean_predecimal_pt1.astype(float), 0.5)
    mean_Plus_500 = np.multiply(mean_predecimal_pt1_Plus_500, 1000)
    upshift_utm = np.subtract(mean_Plus_500, refpts_utm_mean)
    refpts_utm_upshifted = np.add(refpts_utm, upshift_utm)

    # Truncate thousands digits from shifted reference points
    refpts_utm_upshifted_postdecimal = np.mod(refpts_utm_upshifted, 1)
    refpts_utm_upshifted_predecimal = refpts_utm_upshifted.astype(int)
    refpts_utm_upshifted_predecimal_pt1 = np.divide(
        refpts_utm_upshifted_predecimal, 1000
    ).astype(int)
    refpts_utm_upshifted_predecimal_pt1_1row = np.array(
        [
            [
                refpts_utm_upshifted_predecimal_pt1.item(0),
                refpts_utm_upshifted_predecimal_pt1.item(1),
            ]
        ]
    )
    refpts_utm_upshifted_predecimal_pt2 = np.mod(refpts_utm_upshifted_predecimal, 1000)
    refpts_utm_upshifted_disassembled = np.add(
        refpts_utm_upshifted_predecimal_pt2, refpts_utm_upshifted_postdecimal
    )

    # Calculate homography matrix with refpts in pixel coordinates and truncated &
    # shifted refpts in world coordinates
    homography, mask = cv2.findHomography(
        refpts_px, refpts_utm_upshifted_disassembled, cv2.RANSAC, 3.0
    )  # RANSAC: Otulier/Inlier definieren??? # FEHLER:
    log.debug(homography)
    log.debug(mask)

    eval_dict = evaluate_homography(
        refpts_px, refpts_utm_upshifted_disassembled, homography
    )

    # TODO: Prevent different utm zones or hemispheres
    utm_zone = refpts_df["zone_utm"].mode()[0]
    hemisphere = refpts_df["hemisphere"].mode()[0]

    return (
        homography,
        refpts_utm_upshifted_predecimal_pt1_1row,
        upshift_utm,
        utm_zone,
        hemisphere,
        eval_dict,
    )


def evaluate_homography(
    refpts_pixel: np.ndarray,
    refpts_world_upshifted_disassembled: np.ndarray,
    homography_matrix: np.ndarray,
) -> dict:  # TODO: Type hint nested dict during refactoring
    """Calculates transformation error of homography

    Args:
        refpts_pixel (ndarray): Reference points in both pixel and utm coordinates
        refpts_world_upshifted_disassembled (ndarray): Internal variable
        homography_matrix (ndarray): Homography matrix

    Returns:
        dict: Evaluation of transformation error
    """
    # Evaluate accuracy of homography matrix using reference points in world coords
    refpts_pixel_tmp = np.array([refpts_pixel], dtype="float32")
    refpts_world_upshifted_disassembled_transf_3d = cv2.perspectiveTransform(
        refpts_pixel_tmp, homography_matrix
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
    log.debug("Mean transformation error [m]: " + str(eval_df["delta_abs"].mean()))
    log.debug("Maximum transformation error [m]: " + str(eval_df["delta_abs"].max()))
    # sourcery skip: merge-dict-assign
    eval_dict = {}
    eval_dict["mean_transformation_error_m"] = eval_df["delta_abs"].mean()
    eval_dict["Maximum_transformation_error_m"] = eval_df["delta_abs"].max()
    return eval_dict
