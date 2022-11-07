"""
OTVision main module for tracking objects in successive frames of videos
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
from OTVision.helpers.log import log, reset_debug, set_debug

from .iou import track_iou


def main(
    paths: list[Path],
    sigma_l: float = CONFIG["TRACK"]["IOU"]["SIGMA_L"],
    sigma_h: float = CONFIG["TRACK"]["IOU"]["SIGMA_H"],
    sigma_iou: float = CONFIG["TRACK"]["IOU"]["SIGMA_IOU"],
    t_min: int = CONFIG["TRACK"]["IOU"]["T_MIN"],
    t_miss_max: int = CONFIG["TRACK"]["IOU"]["T_MISS_MAX"],
    overwrite: bool = CONFIG["TRACK"]["OVERWRITE"],
    debug: bool = CONFIG["TRACK"]["DEBUG"],
):
    """Read detections from otdet file, perform tracking using iou tracker and
        save tracks to ottrk file.

    Args:
        paths (list[Path]): List of paths to detection files.
        sigma_l (float, optional): Lower confidence threshold. Detections with
            confidences below sigma_l are not even considered for tracking.
            Defaults to CONFIG["TRACK"]["IOU"]["SIGMA_L"].
        sigma_h (float, optional): Upper confidence threshold. Tracks are only
            considered as valid if they contain at least one detection with a confidence
            above sigma_h.
            Defaults to CONFIG["TRACK"]["IOU"]["SIGMA_H"].
        sigma_iou (float, optional): Intersection-Over-Union threshold. Two detections
            in subsequent frames are considered to belong to the same track if their IOU
            value exceeds sigma_iou and this is the highest IOU of all possible
            combination of detections.
            Defaults to CONFIG["TRACK"]["IOU"]["SIGMA_IOU"].
        t_min (int, optional): Minimum number of detections to count as a valid track.
            All tracks with less detections will be dissmissed.
            Defaults to CONFIG["TRACK"]["IOU"]["T_MIN"].
        t_miss_max (int, optional): Maximum number of missed detections before
            continuing a track. If more detections are missing, the track will not be
            continued.
            Defaults to CONFIG["TRACK"]["IOU"]["T_MISS_MAX"].
        overwrite (bool, optional): _description_.
            Defaults to CONFIG["TRACK"]["OVERWRITE"].
        debug (bool, optional):
            _description_. Defaults to CONFIG["TRACK"]["DEBUG"].
    """
    log.info("Start tracking")
    if debug:
        set_debug()

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

            tracks_px = track(
                detections=detections_denormalized,
                sigma_l=sigma_l,
                sigma_h=sigma_h,
                sigma_iou=sigma_iou,
                t_min=t_min,
                t_miss_max=t_miss_max,
            )
            log.info("Detections tracked")

            write(
                tracks_px=tracks_px,
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
    if debug:
        reset_debug()


def track(
    detections: dict,
    sigma_l: float = CONFIG["TRACK"]["IOU"]["SIGMA_L"],
    sigma_h: float = CONFIG["TRACK"]["IOU"]["SIGMA_H"],
    sigma_iou: float = CONFIG["TRACK"]["IOU"]["SIGMA_IOU"],
    t_min: int = CONFIG["TRACK"]["IOU"]["T_MIN"],
    t_miss_max: int = CONFIG["TRACK"]["IOU"]["T_MISS_MAX"],
) -> dict[str, dict]:
    """Perform tracking using track_iou with arguments and add metadata to tracks.

    Args:
        paths (list[Path]): Dict of detections in otdet format.
        sigma_l (float, optional): Lower confidence threshold. Detections with
            confidences below sigma_l are not even considered for tracking.
            Defaults to CONFIG["TRACK"]["IOU"]["SIGMA_L"].
        sigma_h (float, optional): Upper confidence threshold. Tracks are only
            considered as valid if they contain at least one detection with a confidence
            above sigma_h.
            Defaults to CONFIG["TRACK"]["IOU"]["SIGMA_H"].
        sigma_iou (float, optional): Intersection-Over-Union threshold. Two detections
            in subsequent frames are considered to belong to the same track if their IOU
            value exceeds sigma_iou and this is the highest IOU of all possible
            combination of detections.
            Defaults to CONFIG["TRACK"]["IOU"]["SIGMA_IOU"].
        t_min (int, optional): Minimum number of detections to count as a valid track.
            All tracks with less detections will be dissmissed.
            Defaults to CONFIG["TRACK"]["IOU"]["T_MIN"].
        t_miss_max (int, optional): Maximum number of missed detections before
            continuing a track. If more detections are missing, the track will not be
            continued.
            Defaults to CONFIG["TRACK"]["IOU"]["T_MISS_MAX"].
        overwrite (bool, optional): _description_.
            Defaults to CONFIG["TRACK"]["OVERWRITE"].
        debug (bool, optional):
            _description_. Defaults to CONFIG["TRACK"]["DEBUG"].

    Returns:
        dict[str, dict]: Dict of dict of metadata and dict of tracks in ottrk format.
    """
    new_detections = track_iou(
        detections=detections["data"],
        sigma_l=sigma_l,
        sigma_h=sigma_h,
        sigma_iou=sigma_iou,
        t_min=t_min,
        t_miss_max=t_miss_max,
    )

    trk_config = {
        "tracker": "IOU",
        "sigma_l": sigma_l,
        "sigma_h": sigma_h,
        "sigma_iou": sigma_iou,
        "t_min": t_min,
        "t_miss_max": t_miss_max,
    }

    return {
        "metadata": {
            "vid": detections["vid_config"],
            "det": detections["det_config"],
            "trk": trk_config,
        },
        "data": new_detections,
    }


def write(
    tracks_px: dict[str, dict],
    detections_file: Path,
    overwrite: bool = CONFIG["TRACK"]["OVERWRITE"],
):
    """Write or overwrite (or not) tracks using detections file name with different
        suffix.

    Args:
        tracks_px (dict[str, dict]): Dict of tracks including metadata.
        detections_file (Path): File path of the corresponding detections.
        overwrite (bool, optional): Wheter or not to overwrite existing tracks file.
            Defaults to CONFIG["TRACK"]["OVERWRITE"].
    """
    # ?: Check overwrite before tracking instead of before writing tracking?
    # TODO: Export also as csv, trj and alternative json
    tracks_file = Path(detections_file).with_suffix(CONFIG["DEFAULT_FILETYPE"]["TRACK"])
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
