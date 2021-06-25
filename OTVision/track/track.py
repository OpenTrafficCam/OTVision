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
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
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

    new_detections, trajectories_geojson, vehIDs_finished = track_iou(
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

    return tracks_px, trajectories_geojson


# TODO: Implement overwrite as in detect, maybe refactor?
def write(
    tracks_px,
    trajectories_geojson,
    detfile,
    overwrite=CONFIG["TRACK"]["IOU"]["OVERWRITE"],
):
    """
    docstring
    """

    # TODO: #96 Make writing each filetype optional (list of filetypes as parameter)
    # Write JSON
    with open(Path(detfile).with_suffix(".ottrk"), "w") as f:
        json.dump(tracks_px, f, indent=4)
    logging.info("JSON written")

    # Write GeoJSON
    # TODO: #95 Add metadata to GeoJSON (and GPKG)
    with open(Path(detfile).with_suffix(".GeoJSON"), "w") as f:
        json.dump(trajectories_geojson, f, indent=4)
    logging.info("GeoJSON written")

    # Convert to geodataframe and write GPKG
    get_geodataframe_from_framewise_tracks(tracks_px).to_file(
        Path(detfile).with_suffix(".gpkg"), driver="GPKG"
    )
    logging.info("GPKG written")


def get_geodataframe_from_framewise_tracks(tracks_px):
    # TODO: #93 Create LineString GeoDataFrame from Point GeoDataFrame
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

    # aggregate these points with the GrouBy
    # gdf_trajectories = (
    #     (gdf_trajectories.swaplevel().sort_index())
    #     .groupby(["ID"])
    #     .agg(
    #         {
    #             "class": pd.Series.mode,
    #             "conf": "max",
    #             "geometry": pd.Series.apply(
    #                 self=gdf_trajectories["geometry"],
    #                 # BUG: #97 Cannot convert Point to list
    #                 func=lambda x: LineString(x.tolist()),
    #             ),
    #         }
    #     )  # ["geometry"]
    #    .size()  # apply(lambda x: LineString(x.tolist()))
    # )
    # gdf_trajectories = gpd.GeoDataFrame(gdf_trajectories, geometry="geometry")
    print(gdf_trajectories)

    return gdf_trajectories


def get_geodataframe_from_trackwise_tracks(trajectories_px):
    # TODO: #94 Directly create LineString GeoDataFrame
    df_trajectories = pd.DataFrame(trajectories_px)
    df_trajectories = df_trajectories.df.drop(["frames", "bboxes", "conf"], axis=1)
    # gdf_trajectories = gpd.GeoDataFrame(
    #     df_trajectories,
    #     geometry=gpd.points_from_xy(df_trajectories.x, df_trajectories.y),
    # )
    return df_trajectories  # gdf_trajectories


def main(paths=CONFIG["TESTDATAFOLDER"], config_track=config_track_default):
    """
    docstring
    """
    filetypes = ".otdet"
    detections_files = get_files(paths, filetypes)
    for detections_file in detections_files:
        logging.info(f"New detections file: {detections_file}")
        detections, dir, filename = read(detections_file)
        logging.info("detections read")
        detections = denormalize(detections)
        logging.info("detections denormalized")
        tracks_px, trajectories_geojson = track(detections)
        logging.info("detections tracked")
        write(tracks_px, trajectories_geojson, detections_file)
        logging.info("Tracking finished")


# To Dos:
# - Improve memory management (MemoryError after two 2h detections)
# - Perform logging by dedicated python package
