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


from pathlib import Path
from typing import Tuple

from OTVision.config import (
    CONFIG,
    DEBUG,
    DEFAULT_FILETYPE,
    DETECT,
    FILETYPES,
    IOU,
    OVERWRITE,
    SIGMA_H,
    SIGMA_IOU,
    SIGMA_L,
    T_MIN,
    T_MISS_MAX,
    TRACK,
)
from OTVision.dataformat import DATA, DETECTIONS, METADATA
from OTVision.helpers.files import (
    _check_and_update_metadata_inplace,
    denormalize_bbox,
    get_files,
    write_json,
)
from OTVision.helpers.log import log, reset_debug, set_debug
from OTVision.track.preprocess import Preprocess, Splitter

from .iou import track_iou


def main(
    paths: list[Path],
    sigma_l: float = CONFIG[TRACK][IOU][SIGMA_L],
    sigma_h: float = CONFIG[TRACK][IOU][SIGMA_H],
    sigma_iou: float = CONFIG[TRACK][IOU][SIGMA_IOU],
    t_min: int = CONFIG[TRACK][IOU][T_MIN],
    t_miss_max: int = CONFIG[TRACK][IOU][T_MISS_MAX],
    overwrite: bool = CONFIG[TRACK][OVERWRITE],
    debug: bool = CONFIG[TRACK][DEBUG],
) -> None:
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
        overwrite (bool, optional): Whether or not to overwrite existing tracks files.
            Defaults to CONFIG["TRACK"]["OVERWRITE"].
        debug (bool, optional): Whether or not to run in debug mode.
            Defaults to CONFIG["TRACK"]["DEBUG"].
    """
    log.info("Start tracking")
    if debug:
        set_debug()

    filetypes = CONFIG[FILETYPES][DETECT]
    detections_files = get_files(paths=paths, filetypes=filetypes)

    if not detections_files:
        raise FileNotFoundError(f"No files of type '{filetypes}' found to track!")

    preprocessor = Preprocess()
    preprocessed = preprocessor.run(detections_files)

    file_type = CONFIG[DEFAULT_FILETYPE][TRACK]
    for frame_group in preprocessed.frame_groups:
        existing_output_files = frame_group.get_existing_output_files(
            with_suffix=file_type
        )

        if not overwrite and (len(existing_output_files) > 0):
            log.warning(
                (
                    f"{existing_output_files} already exist(s)."
                    "To overwrite, set overwrite to True"
                )
            )
            continue

        metadata = preprocessed.metadata
        detections = frame_group.to_dict()
        _check_and_update_metadata_inplace(otdict=detections)

        detections_denormalized = denormalize_bbox(detections, metadata=metadata)

        tracks_px, new_metadata = track(
            detections=detections_denormalized,
            sigma_l=sigma_l,
            sigma_h=sigma_h,
            sigma_iou=sigma_iou,
            t_min=t_min,
            t_miss_max=t_miss_max,
            metadata=metadata,
        )
        log.info(f"Successfully tracked {frame_group.order_key}")

        # Split into files of group
        splitted: dict[str, list[dict]] = Splitter().split(tracks_px)
        for file_path, serializable_detections in splitted.items():
            output = build_output(file_path, serializable_detections, new_metadata)
            write_json(
                dict_to_write=output,
                file=Path(file_path),
                filetype=file_type,
                overwrite=overwrite,
            )

    if debug:
        reset_debug()


def track(
    detections: dict,  # TODO: Type hint nested dict during refactoring
    sigma_l: float = CONFIG[TRACK][IOU][SIGMA_L],
    sigma_h: float = CONFIG[TRACK][IOU][SIGMA_H],
    sigma_iou: float = CONFIG[TRACK][IOU][SIGMA_IOU],
    t_min: int = CONFIG[TRACK][IOU][T_MIN],
    t_miss_max: int = CONFIG[TRACK][IOU][T_MISS_MAX],
    metadata: dict[str, dict] = {},
) -> Tuple[
    dict[str, dict], dict[str, dict]
]:  # TODO: Type hint nested dict during refactoring
    """Perform tracking using track_iou with arguments and add metadata to tracks.

    Args:
        detections (dict): Dict of detections in .otdet format.
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
        metadata (dict[str, dict]): dict of metadata per input file.

    Returns:
        dict[str, dict]: Dict of tracks in ottrk format.
        dict[str, dict]: Dict of metadata
    """

    new_detections = track_iou(
        detections=detections["data"],
        sigma_l=sigma_l,
        sigma_h=sigma_h,
        sigma_iou=sigma_iou,
        t_min=t_min,
        t_miss_max=t_miss_max,
    )

    for id in metadata:
        metadata[id]["trk"] = {
            "tracker": "IOU",
            "sigma_l": sigma_l,
            "sigma_h": sigma_h,
            "sigma_iou": sigma_iou,
            "t_min": t_min,
            "t_miss_max": t_miss_max,
        }

    log.info("Detections tracked")

    return new_detections, metadata


def build_output(
    file_path: str, detections: list[dict], metadata: dict[str, dict]
) -> dict:
    return {METADATA: metadata[file_path], DATA: {DETECTIONS: detections}}
