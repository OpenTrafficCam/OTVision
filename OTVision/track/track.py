"""
OTVision main module for tracking objects in successive frames of videos
"""

# points and transform tracksectory points from pixel into world coordinates.

import logging
import uuid

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
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterator

from tqdm import tqdm

from OTVision import dataformat
from OTVision.config import (
    CONFIG,
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
from OTVision.dataformat import DATA, DETECTIONS, FINISHED, METADATA, TRACK_ID
from OTVision.helpers.files import denormalize_bbox, get_files, write_json
from OTVision.helpers.log import LOGGER_NAME
from OTVision.track.preprocess import Preprocess, PreprocessOld, Splitter

from .iou import id_generator, track_iou

log = logging.getLogger(LOGGER_NAME)

IdGenerator = Callable[[], str]


RESULT_DETECTIONS = "Detections"
RESULT_FRAME_GROUP_ID = "FrameGroupId"
RESULT_ACTIVE_TRACKS = "ActiveTrackIds"
RESULT_TRACK_IDS = "TrackIds"


def main_old(
    paths: list[Path],
    sigma_l: float = CONFIG[TRACK][IOU][SIGMA_L],
    sigma_h: float = CONFIG[TRACK][IOU][SIGMA_H],
    sigma_iou: float = CONFIG[TRACK][IOU][SIGMA_IOU],
    t_min: int = CONFIG[TRACK][IOU][T_MIN],
    t_miss_max: int = CONFIG[TRACK][IOU][T_MISS_MAX],
    overwrite: bool = CONFIG[TRACK][OVERWRITE],
    tracking_run_id_generator: IdGenerator = lambda: str(uuid.uuid4()),
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
        tracking_run_id_generator (IdGenerator): Generator used to create a unique id
            for this tracking run
    """

    filetypes = CONFIG[FILETYPES][DETECT]
    detections_files = get_files(paths=paths, filetypes=filetypes)

    start_msg = f"Start tracking of {len(detections_files)} detections files"
    log.info(start_msg)
    print(start_msg)

    if not detections_files:
        raise FileNotFoundError(f"No files of type '{filetypes}' found to track!")

    tracking_run_id = tracking_run_id_generator()
    preprocessor = PreprocessOld()
    preprocessed = preprocessor.run(detections_files)

    file_type = CONFIG[DEFAULT_FILETYPE][TRACK]
    for frame_group_id, frame_group in tqdm(
        enumerate(preprocessed.frame_groups),
        desc="Tracked frame group",
        unit="framegroup",
    ):
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

        log.info(f"Track {str(frame_group.order_key)}")

        metadata = preprocessed.metadata
        detections = frame_group.to_dict()
        tracker_data: dict = {
            dataformat.NAME: "IOU",
            dataformat.SIGMA_L: sigma_l,
            dataformat.SIGMA_H: sigma_h,
            dataformat.SIGMA_IOU: sigma_iou,
            dataformat.T_MIN: t_min,
            dataformat.T_MISS_MAX: t_miss_max,
        }
        frame_group.update_metadata(metadata, tracker_data)

        detections_denormalized = denormalize_bbox(detections, metadata=metadata)

        tracks_px, _, last_det_frame = track(
            detections=detections_denormalized,
            sigma_l=sigma_l,
            sigma_h=sigma_h,
            sigma_iou=sigma_iou,
            t_min=t_min,
            t_miss_max=t_miss_max,
        )

        mark_last_detections_as_finished(
            tracks_px, last_det_frame, set(last_det_frame.keys())
        )

        log.debug(f"Successfully tracked {frame_group.order_key}")

        # Split into files of group
        tracks_per_file: dict[str, list[dict]] = Splitter().split(tracks_px)
        for file_path, serializable_detections in tracks_per_file.items():
            output = build_output(
                file_path,
                serializable_detections,
                metadata,
                tracking_run_id,
                frame_group_id,
            )
            write_json(
                dict_to_write=output,
                file=Path(file_path),
                filetype=file_type,
                overwrite=overwrite,
            )

        log.info(f"Successfully tracked and wrote {frame_group.order_key}")

    finished_msg = "Finished tracking"
    log.info(finished_msg)
    print(finished_msg)


def main(
    paths: list[Path],
    sigma_l: float = CONFIG[TRACK][IOU][SIGMA_L],
    sigma_h: float = CONFIG[TRACK][IOU][SIGMA_H],
    sigma_iou: float = CONFIG[TRACK][IOU][SIGMA_IOU],
    t_min: int = CONFIG[TRACK][IOU][T_MIN],
    t_miss_max: int = CONFIG[TRACK][IOU][T_MISS_MAX],
    overwrite: bool = CONFIG[TRACK][OVERWRITE],
    tracking_run_id_generator: IdGenerator = lambda: str(uuid.uuid4()),
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
        tracking_run_id_generator (IdGenerator): Generator used to create a unique id
            for this tracking run
    """

    filetypes = CONFIG[FILETYPES][DETECT]
    detections_files = get_files(paths=paths, filetypes=filetypes)

    start_msg = f"Start tracking of {len(detections_files)} detections files"
    log.info(start_msg)
    print(start_msg)

    if not detections_files:
        raise FileNotFoundError(f"No files of type '{filetypes}' found to track!")

    tracking_run_id = tracking_run_id_generator()
    preprocessor_old = PreprocessOld()

    preprocessor = Preprocess()
    preprocessed = preprocessor.run(detections_files)

    file_type = CONFIG[DEFAULT_FILETYPE][TRACK]

    for frame_group_id, frame_range in tqdm(
        enumerate(preprocessed),
        desc="Tracked frame ranges",
        unit="framerange",
    ):
        print()
        print(f"Process frame group {frame_group_id}")

        tracker_data: dict = {
            dataformat.NAME: "IOU",
            dataformat.SIGMA_L: sigma_l,
            dataformat.SIGMA_H: sigma_h,
            dataformat.SIGMA_IOU: sigma_iou,
            dataformat.T_MIN: t_min,
            dataformat.T_MISS_MAX: t_miss_max,
        }
        frame_range.update_metadata(tracker_data)

        vehicle_id_generator = id_generator()

        previous_active_tracks: list = []
        last_track_frame: dict[int, int] = {}
        result_store: dict[Path, dict] = {}
        frame_offset = 0

        for file_path in frame_range.files:
            print(f"Process file {file_path} in frame group {frame_group_id}")
            preprocessed_old = preprocessor_old.run([file_path], frame_offset)
            frame_group = preprocessed_old.frame_groups[0]  # should be exactly one FG

            frame_offset = frame_group.frames[-1].frame + 1

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

            log.info(f"Track {str(frame_group.order_key)}")

            metadata = frame_range.metadata_for(file_path)

            detections = frame_group.to_dict()
            detections_denormalized = denormalize_bbox(detections, metadata=metadata)

            tracks_px, previous_active_tracks, last_frame_update = track(
                detections=detections_denormalized,
                sigma_l=sigma_l,
                sigma_h=sigma_h,
                sigma_iou=sigma_iou,
                t_min=t_min,
                t_miss_max=t_miss_max,
                previous_active_tracks=previous_active_tracks,
                vehicle_id_generator=vehicle_id_generator,
            )
            print("remaining active tracks", len(previous_active_tracks))
            last_track_frame.update(last_frame_update)
            log.debug(f"Successfully tracked {frame_group.order_key}")

            # store results of iou and mark vehIDs yet to be finished
            result_store[file_path] = {
                RESULT_DETECTIONS: tracks_px,
                RESULT_FRAME_GROUP_ID: frame_group_id,
                RESULT_ACTIVE_TRACKS: set(last_frame_update.keys()),
                RESULT_TRACK_IDS: set(last_frame_update.keys()),
            }

            active_track_ids = {t[TRACK_ID] for t in previous_active_tracks}
            # update missing tracks of previous files
            to_delete = set()
            for det_file, results in result_store.items():
                active_tracks = results[RESULT_ACTIVE_TRACKS]

                active_update = {
                    vehID for vehID in active_track_ids if vehID in active_tracks
                }

                results[RESULT_ACTIVE_TRACKS] = active_update
                if len(active_update) == 0:
                    track_ids = results[RESULT_TRACK_IDS]
                    mark_and_write_results(
                        results, last_track_frame, track_ids, metadata, tracking_run_id
                    )
                    to_delete.add(det_file)

            for det_file in to_delete:
                del result_store[det_file]

        # write last files of frame group
        # even though some tracks mights still be active
        for det_file, results in result_store.items():
            track_ids = results[RESULT_TRACK_IDS]
            mark_and_write_results(
                results, last_track_frame, track_ids, metadata, tracking_run_id
            )

        del result_store

        log.info("Successfully tracked and wrote ")  # TODO ord key for frame range

    finished_msg = "Finished tracking"
    log.info(finished_msg)
    print(finished_msg)


def mark_and_write_results(
    results: dict,
    last_track_frame: dict,
    track_ids: set,
    metadata: dict,
    tracking_run_id: str,
) -> None:
    # no active tracks remaining, so last track frame metadata
    # should be correct for all contained tracks,
    # thus set finished flags now
    mark_last_detections_as_finished(
        detections=results[RESULT_DETECTIONS],
        last_track_frame=last_track_frame,
        track_ids=track_ids,
    )

    # write marked detections to track file and delete the data
    write_track_file(
        tracks_px=results[RESULT_DETECTIONS],
        metadata=metadata,
        tracking_run_id=tracking_run_id,
        frame_group_id=results[RESULT_FRAME_GROUP_ID],
    )
    del results[RESULT_DETECTIONS]


def write_track_file(
    tracks_px: dict, metadata: dict, tracking_run_id: str, frame_group_id: int
) -> None:
    # Split into files of group
    file_type = CONFIG[DEFAULT_FILETYPE][TRACK]
    tracks_per_file: dict[str, list[dict]] = Splitter().split(tracks_px)
    for file_path, serializable_detections in tracks_per_file.items():
        output = build_output(
            file_path,
            serializable_detections,
            metadata,
            tracking_run_id,
            frame_group_id,
        )
        write_json(
            dict_to_write=output,
            file=Path(file_path),
            filetype=file_type,
            overwrite=True,  # TODO
        )

        log.info(f"Successfully tracked and wrote {file_path}")


def track(
    detections: dict,  # TODO: Type hint nested dict during refactoring
    sigma_l: float = CONFIG[TRACK][IOU][SIGMA_L],
    sigma_h: float = CONFIG[TRACK][IOU][SIGMA_H],
    sigma_iou: float = CONFIG[TRACK][IOU][SIGMA_IOU],
    t_min: int = CONFIG[TRACK][IOU][T_MIN],
    t_miss_max: int = CONFIG[TRACK][IOU][T_MISS_MAX],
    previous_active_tracks: list = [],
    vehicle_id_generator: Iterator[int] = id_generator(),
) -> tuple[
    dict[str, dict], list, dict[int, int]
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
        #TODO

    Returns:
        tuple[dict[int, dict], list]: Dict of tracks in ottrk format
            and list of active tracks (iou format?).
    """

    new_detections, active_tracks, last_track_frame = track_iou(
        detections=detections[DATA],
        sigma_l=sigma_l,
        sigma_h=sigma_h,
        sigma_iou=sigma_iou,
        t_min=t_min,
        t_miss_max=t_miss_max,
        previous_active_tracks=previous_active_tracks,
        vehicle_id_generator=vehicle_id_generator,
    )
    log.info("Detections tracked")

    return new_detections, active_tracks, last_track_frame


def mark_last_detections_as_finished(
    detections: dict[str, dict[int, dict]],
    last_track_frame: dict[int, int],
    track_ids: set[int],
) -> None:
    # invert last occurrence frame dict
    frame_ending_tracks = defaultdict(set)
    for vehID in track_ids:
        frame_ending_tracks[last_track_frame[vehID]].add(vehID)

    for frame_num, frame_det in tqdm(
        detections.items(), desc="Mark finished frames", unit="frames"
    ):
        for ending_track in frame_ending_tracks[int(frame_num)]:
            frame_det[ending_track][FINISHED] = True
            del last_track_frame[ending_track]


def build_output(
    file_path: str,
    detections: list[dict],
    metadata: dict[str, dict],
    tracking_run_id: str,
    frame_group_id: int,
) -> dict:
    metadata[file_path][dataformat.TRACKING][
        dataformat.TRACKING_RUN_ID
    ] = tracking_run_id
    metadata[file_path][dataformat.TRACKING][dataformat.FRAME_GROUP] = frame_group_id
    return {METADATA: metadata[file_path], DATA: {DETECTIONS: detections}}
