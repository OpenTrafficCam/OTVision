"""
OTVision main module for tracking objects in successive frames of videos.
"""
# points and transform tracksectory points from pixel into world coordinates.

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
# GNU General Public License for more detectionsails.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import json
from pathlib import Path

from OTVision.config import CONFIG
from OTVision.helpers.files import denormalize, get_files
from OTVision.helpers.log import log

from .iou import track_iou


def main(
    paths,
    yolo_mode="spp",  # Why yolo mode?
    sigma_l=CONFIG["TRACK"]["IOU"]["SIGMA_L"],
    sigma_h=CONFIG["TRACK"]["IOU"]["SIGMA_H"],
    sigma_iou=CONFIG["TRACK"]["IOU"]["SIGMA_IOU"],
    t_min=CONFIG["TRACK"]["IOU"]["T_MIN"],
    t_miss_max=CONFIG["TRACK"]["IOU"]["T_MISS_MAX"],
    overwrite=CONFIG["TRACK"]["OVERWRITE"],
    debug: bool = CONFIG["TRACK"]["DEBUG"],
):
    log.info("Start tracking")
    if debug:
        log.setLevel("DEBUG")
        log.debug("Logging track.main() in debug mode")

    filetype = CONFIG["DEFAULT_FILETYPE"]["DETECT"]
    detections_files = get_files(paths, filetype)
    for detections_file in detections_files:
        log.info(f"Try tracking {detections_file}")

        try:
            with open(detections_file) as f:
                detections = json.load(f)
            log.info(f"{filetype} read")

            detections_denormalized = denormalize(detections)
            log.info("Detections denormalized")

            tracks_px, trajectories_geojson = track(
                detections=detections_denormalized,
                yolo_mode=yolo_mode,
                sigma_l=sigma_l,
                sigma_h=sigma_h,
                sigma_iou=sigma_iou,
                t_min=t_min,
                t_miss_max=t_miss_max,
            )
            log.info("Detections tracked")

            write(
                tracks_px=tracks_px,
                trajectories_geojson=trajectories_geojson,
                detections_file=detections_file,
                overwrite=overwrite,
            )
        except OSError as oe:
            log.error(
                (
                    f'Could not open "{detections_file}". '
                    f"Following exception occured: {str(oe)}"
                )
            )
        except json.JSONDecodeError as je:
            log.error(
                (
                    f'Unable to decode "{detections_file}" as JSON.'
                    f"Following exception occured: {str(je)}"
                )
            )


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

    trk_config = {
        "yolo_mode": yolo_mode,
        "tracker": "IOU",
        "sigma_l": sigma_l,
        "sigma_h": sigma_h,
        "sigma_iou": sigma_iou,
        "t_min": t_min,
        "t_miss_max": t_miss_max,
    }

    tracks_px = {
        "vid_config": detections["vid_config"],
        "det_config": detections["det_config"],
        "trk_config": trk_config,
        "data": new_detections,
    }

    return tracks_px, trajectories_geojson


# TODO: Implement overwrite as in detect, maybe refactor?
def write(
    tracks_px,
    detections_file,
    overwrite=CONFIG["TRACK"]["OVERWRITE"],
):
    # ?: Check overwrite before tracking instead of before writing tracking?
    # TODO: Export also as csv, trj and alternative json
    tracks_file = Path(detections_file).with_suffix(CONFIG["FILETYPES"]["TRACK"])
    tracks_file_already_exists = tracks_file.is_file()
    if overwrite or not tracks_file_already_exists:
        # Write JSON
        with open(tracks_file, "w") as f:
            json.dump(tracks_px, f, indent=4)
        if tracks_file_already_exists:
            log.info(f"{tracks_file} overwritten")
        else:
            log.info(f"{tracks_file}  file written")
    else:
        log.info(f"{tracks_file} already exists. To overwrite, set overwrite=True")
