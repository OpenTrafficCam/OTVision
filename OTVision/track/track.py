# OTVision: Python module to calculate homography matrix from reference
# points and transform tracksectory points from pixel into world coordinates.

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
# GNU General Public License for more detectionsails.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import geopandas as gpd
from config import CONFIG
from track.iou import track_iou
from helpers.files import get_files, denormalize

# TODO:Change structure and naming to according to detect
config_track_default = {
    "yolo_mode": "spp",
    "sigma_l": 0.1,
    "sigma_h": 0.85,
    "sigma_iou": 0.4,
    "t_min": 12,
    "save_age": 5,
    "overwrite": True,
}


def read(detections_file):
    """
    docstring
    """
    dir = Path(detections_file).parent
    filename = Path(detections_file).stem.rsplit("_", 1)[0]
    # detections_suffix = Path(detections_file).stem.rsplit("_", 1)[1]
    filetype = Path(detections_file).suffix
    if filetype == ".otdet":
        with open(detections_file) as f:
            detections = json.load(f)
    else:
        raise ValueError("Filetype " + filetype + " cannot be read, has to be .otdet")
    return detections, dir, filename


def track(detections, trk_config=config_track_default):
    """
    docstring
    """

    new_detections, tracks_finished, vehIDs_finished = track_iou(
        detections["data"],
        trk_config["sigma_l"],
        trk_config["sigma_h"],
        trk_config["sigma_iou"],
        trk_config["t_min"],
        trk_config["save_age"],
    )
    trk_config["tracker"] = "IOU"
    tracks_px = {}
    tracks_px["vid_config"] = detections["vid_config"]
    tracks_px["det_config"] = detections["det_config"]
    tracks_px["trk_config"] = trk_config
    tracks_px["data"] = new_detections

    return tracks_px


# TODO: Implement overwrite as in detect, maybe refactor?
def write(tracks_px, detfile, overwrite=CONFIG["TRACK"]["IOU"]["OVERWRITE"]):
    """
    docstring
    """

    # Write to json
    with open(Path(detfile).with_suffix(".ottrk"), "w") as f:
        json.dump(tracks_px, f, indent=4)

    # Convert to geodataframe and write to gpkg
    get_geodataframe(tracks_px).to_file(
        Path(detfile).with_suffix(".otgpkg"), driver="GPKG"
    )


def get_geodataframe(tracks_px):
    df_trajectories = pd.DataFrame.from_dict(
        {
            (frame_nr, det_nr): tracks_px["data"][frame_nr][det_nr]
            for frame_nr in tracks_px["data"].keys()
            for det_nr in tracks_px["data"][frame_nr].keys()
        },
        orient="index",
    ).rename_axis(("frame", "ID"))
    gdf_trajectories = gpd.GeoDataFrame(
        df_trajectories,
        geometry=gpd.points_from_xy(df_trajectories.x, df_trajectories.y),
    )
    return gdf_trajectories


def main(paths, config_track=config_track_default):
    """
    docstring
    """
    filetypes = ".otdet"
    detections_files = get_files(paths, filetypes)
    for detections_file in detections_files:
        print(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            + ": New detections file: "
            + detections_file
        )
        detections, dir, filename = read(detections_file)
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ": detections read")
        detections = denormalize(detections)
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ": detections denormalize")
        tracks_px = track(detections)
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ": detections tracked")
        write(tracks_px, detections_file)
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ": Tracks written")


# To Dos:
# - Improve memory management (MemoryError after two 2h detections)
# - Perform logging by dedicated python package
