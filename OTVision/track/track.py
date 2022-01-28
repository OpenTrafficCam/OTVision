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

from OTVision.config import CONFIG
from .iou import track_iou
from OTVision.helpers.files import get_files, denormalize


def main(
    paths,
    yolo_mode="spp",  # Why yolo mode?
    sigma_l=CONFIG["TRACK"]["IOU"]["SIGMA_L"],
    sigma_h=CONFIG["TRACK"]["IOU"]["SIGMA_H"],
    sigma_iou=CONFIG["TRACK"]["IOU"]["SIGMA_IOU"],
    t_min=CONFIG["TRACK"]["IOU"]["T_MIN"],
    t_miss_max=CONFIG["TRACK"]["IOU"]["T_MISS_MAX"],
    overwrite=CONFIG["TRACK"]["IOU"]["OVERWRITE"],
):
    filetypes = CONFIG["DEFAULT_FILETYPE"]["DETECT"]
    detections_files = get_files(paths, filetypes)
    for detections_file in detections_files:
        logging.info(f"New detections file: {detections_file}")
        try:
            with open(detections_file) as f:
                detections = json.load(f)
        except:
            logging.error(f"Could not read {detections_file} as json")
            continue
        logging.info("detections read")
        detections = denormalize(detections)
        logging.info("detections denormalized")
        tracks_px, trajectories_geojson = track(
            detections=detections,
            yolo_mode=yolo_mode,
            sigma_l=sigma_l,
            sigma_h=sigma_h,
            sigma_iou=sigma_iou,
            t_min=t_min,
            t_miss_max=t_miss_max,
        )
        logging.info("detections tracked")
        write(
            tracks_px=tracks_px,
            trajectories_geojson=trajectories_geojson,
            detections_file=detections_file,
        )
        logging.info("Tracking finished")


def track(
    detections,
    yolo_mode="spp",
    sigma_l=CONFIG["TRACK"]["IOU"]["SIGMA_L"],
    sigma_h=CONFIG["TRACK"]["IOU"]["SIGMA_H"],
    sigma_iou=CONFIG["TRACK"]["IOU"]["SIGMA_IOU"],
    t_min=CONFIG["TRACK"]["IOU"]["T_MIN"],
    t_miss_max=CONFIG["TRACK"]["IOU"]["T_MISS_MAX"],
):
    new_detections, trajectories_geojson, vehIDs_finished = track_iou(
        detections=detections["data"],
        sigma_l=sigma_l,
        sigma_h=sigma_h,
        sigma_iou=sigma_iou,
        t_min=t_min,
        t_miss_max=t_miss_max,
    )
    trk_config = {}
    trk_config["yolo_mode"] = yolo_mode
    trk_config["tracker"] = "IOU"
    trk_config["sigma_l"] = sigma_l
    trk_config["sigma_h"] = sigma_h
    trk_config["sigma_iou"] = sigma_iou
    trk_config["t_min"] = t_min
    trk_config["t_miss_max"] = t_miss_max
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
    detections_file,
    overwrite=CONFIG["TRACK"]["IOU"]["OVERWRITE"],
):
    # TODO: #96 Make writing each filetype optional (list of filetypes as parameter)
    # Write JSON
    with open(Path(detections_file).with_suffix(".ottrk"), "w") as f:
        json.dump(tracks_px, f, indent=4)
    logging.info("JSON written")

    # Write GeoJSON
    # TODO: #95 Add metadata to GeoJSON (and GPKG)
    with open(Path(detections_file).with_suffix(".GeoJSON"), "w") as f:
        json.dump(trajectories_geojson, f, indent=4)
    logging.info("GeoJSON written")

    # Convert to geodataframe and write GPKG
    get_geodataframe_from_framewise_tracks(tracks_px).to_file(
        Path(detections_file).with_suffix(".gpkg"), driver="GPKG"
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


# To Dos:
# - Improve memory management (MemoryError after two 2h detections)
# - Perform logging by dedicated python package
