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
from OTVision.track.preprocess import (
    FrameChunk,
    FrameChunkParser,
    FrameIndexer,
    Preprocess,
)

from .iou import TrackedDetections, TrackingResult, id_generator, track_iou

log = logging.getLogger(LOGGER_NAME)

IdGenerator = Callable[[], str]


class TrackedChunk(TrackedDetections):
    """Tracking results combined with respective metadata."""

    def __init__(
        self,
        detections: TrackedDetections,
        file_path: Path,
        frame_group_id: int,
        frame_offset: int,
        metadata: dict,
    ) -> None:
        super().__init__(
            detections._detections,
            detections._detected_ids,
            detections._active_track_ids,
        )
        self.file_path = file_path
        self.frame_offset = frame_offset
        self._frame_group_id = frame_group_id
        self.metadata = metadata


class TrackingResultStore:
    """TrackingResultStore manages TrackedChunks
    and data required for the next tracking iteration:
    - the remaining active tracks of the last iteration
    - a global lookup table for each tracks last detection frame
    """

    def __init__(self, tracking_run_id: str) -> None:
        self.tracking_run_id = tracking_run_id
        self._tracked_chunks: list[TrackedChunk] = list()
        self.last_track_frame: dict[int, int] = dict()
        self.active_tracks: list[dict] = []

    def store_tracking_result(
        self,
        tracking_result: TrackingResult,
        file_path: Path,
        frame_group_id: int,
        frame_offset: int,
        metadata: dict,
    ) -> None:
        # store tracking results
        self._tracked_chunks.append(
            TrackedChunk(
                detections=tracking_result.tracked_detections,
                file_path=file_path,
                frame_group_id=frame_group_id,
                frame_offset=frame_offset,
                metadata=metadata,
            )
        )
        # update global lookup table
        self.last_track_frame.update(tracking_result.last_track_frame)

        # replace last iterations active tracks
        self.active_tracks = tracking_result.active_tracks
        print("remaining active tracks", len(self.active_tracks))

        # collect remaining active ids and update tracked chunks
        new_active_ids = {t[TRACK_ID] for t in self.active_tracks}
        for result in self._tracked_chunks:
            result.update_active_track_ids(new_active_ids)

    def get_finished_results(self) -> list[TrackedChunk]:
        """Return all TrackedChunks that have no remaining active track ids.
        These are ready to be written to a result file.

        Returns:
            list[TrackedChunk]: finished TrackChunks
        """
        finished = [chunk for chunk in self._tracked_chunks if chunk.is_finished()]

        # remove finished TrackedChunks from manager
        self._tracked_chunks = [
            chunk for chunk in self._tracked_chunks if chunk not in finished
        ]

        return finished


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
        log.warning(f"No files of type '{filetypes}' found to track!")
        return

    tracking_run_id = tracking_run_id_generator()
    preprocessor = Preprocess()
    preprocessed = preprocessor.run(detections_files)
    file_type = CONFIG[DEFAULT_FILETYPE][TRACK]

    for frame_group_id, frame_group in tqdm(
        enumerate(preprocessed),
        desc="Tracked frame groups",
        unit=" framegroup",
    ):
        print()
        print(f"Process frame group {frame_group_id}")

        # update metadata for all files in FrameGroup
        frame_group.update_metadata(
            tracker_metadata(sigma_l, sigma_h, sigma_iou, t_min, t_miss_max)
        )

        # metadata and tracking results to keep track during iterations
        # within a FrameGroups track/vehicle and frame ids should be unique
        vehicle_id_generator = id_generator()
        frame_offset = 0
        track_result_store = TrackingResultStore(tracking_run_id)

        # process each otdet file in frame group
        for file_path in frame_group.files:
            print(f"Process file {file_path} in frame group {frame_group_id}")

            # read detection data
            chunk = FrameChunkParser.parse(file_path, frame_offset)

            if skip_existing_output_files(chunk, overwrite, file_type):
                continue

            log.info(f"Track {str(chunk)}")

            detections = chunk.to_dict()
            detections_denormalized = denormalize_bbox(
                detections, metadata=frame_group._files_metadata
            )

            tracking_result = track(
                detections=detections_denormalized,
                sigma_l=sigma_l,
                sigma_h=sigma_h,
                sigma_iou=sigma_iou,
                t_min=t_min,
                t_miss_max=t_miss_max,
                previous_active_tracks=track_result_store.active_tracks,
                vehicle_id_generator=vehicle_id_generator,
            )

            track_result_store.store_tracking_result(
                tracking_result,
                file_path=file_path,
                frame_group_id=frame_group_id,
                frame_offset=frame_offset,
                metadata=frame_group.metadata_for(file_path),
            )
            log.debug(f"Successfully tracked {chunk}")
            frame_offset = chunk.last_frame_id() + 1

            for finished_chunk in track_result_store.get_finished_results():
                mark_and_write_results(finished_chunk, track_result_store, overwrite)

        # write last files of frame group
        # even though some tracks mights still be active
        for finished_chunk in track_result_store._tracked_chunks:
            mark_and_write_results(finished_chunk, track_result_store, overwrite)

        log.info("Successfully tracked and wrote ")

    finished_msg = "Finished tracking"
    log.info(finished_msg)
    print(finished_msg)


def skip_existing_output_files(
    chunk: FrameChunk, overwrite: bool, file_type: str
) -> bool:
    existing_output_files = chunk.get_existing_output_files(with_suffix=file_type)

    if not overwrite and (len(existing_output_files) > 0):
        log.warning(
            (
                f"{existing_output_files} already exist(s)."
                "To overwrite, set overwrite to True"
            )
        )
        return True

    return False


def tracker_metadata(
    sigma_l: float, sigma_h: float, sigma_iou: float, t_min: float, t_miss_max: float
) -> dict:
    return {
        dataformat.NAME: "IOU",
        dataformat.SIGMA_L: sigma_l,
        dataformat.SIGMA_H: sigma_h,
        dataformat.SIGMA_IOU: sigma_iou,
        dataformat.T_MIN: t_min,
        dataformat.T_MISS_MAX: t_miss_max,
    }


def mark_and_write_results(
    chunk: TrackedChunk,
    result_store: TrackingResultStore,
    overwrite: bool,
) -> None:
    # no active tracks remaining, so last track frame metadata
    # should be correct for all contained tracks,
    # thus set finished flags now
    mark_last_detections_as_finished(chunk, result_store)

    # write marked detections to track file and delete the data
    file_path = chunk.file_path

    # reindex frames before writing ottrk file
    file_type = CONFIG[DEFAULT_FILETYPE][TRACK]
    serializable_detections: list[dict] = FrameIndexer().reindex(
        chunk._detections, frame_offset=chunk.frame_offset
    )

    output = build_output(
        serializable_detections,
        chunk.metadata,
        result_store.tracking_run_id,
        chunk._frame_group_id,
    )
    write_json(
        dict_to_write=output,
        file=Path(file_path),
        filetype=file_type,
        overwrite=overwrite,
    )

    log.info(f"Successfully tracked and wrote {file_path}")
    del chunk._detections


def track(
    detections: dict,  # TODO: Type hint nested dict during refactoring
    sigma_l: float = CONFIG[TRACK][IOU][SIGMA_L],
    sigma_h: float = CONFIG[TRACK][IOU][SIGMA_H],
    sigma_iou: float = CONFIG[TRACK][IOU][SIGMA_IOU],
    t_min: int = CONFIG[TRACK][IOU][T_MIN],
    t_miss_max: int = CONFIG[TRACK][IOU][T_MISS_MAX],
    previous_active_tracks: list = [],
    vehicle_id_generator: Iterator[int] = id_generator(),
) -> TrackingResult:  # TODO: Type hint nested dict during refactoring
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
        previous_active_tracks (list): a list of remaining active tracks
            from previous iterations.
        vehicle_id_generator (Iterator[int]): provides ids for new tracks

    Returns:
        TrackingResult: A result object holding tracked detecktions in ottrk format
            and list of active tracks (iou format?)
            and a lookup table for each tracks last detection frame.
    """

    result = track_iou(
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

    return result


def mark_last_detections_as_finished(
    chunk: TrackedChunk, result_store: TrackingResultStore
) -> None:
    # invert last occurrence frame dict
    last_track_frame = result_store.last_track_frame

    frame_ending_tracks = defaultdict(set)
    for vehID in chunk._detected_ids:
        frame_ending_tracks[last_track_frame[vehID]].add(vehID)

    for frame_num, frame_det in chunk._detections.items():
        for ending_track in frame_ending_tracks[int(frame_num)]:
            frame_det[ending_track][FINISHED] = True
            del last_track_frame[ending_track]


def build_output(
    detections: list[dict],
    metadata: dict,
    tracking_run_id: str,
    frame_group_id: int,
) -> dict:
    metadata[dataformat.TRACKING][dataformat.TRACKING_RUN_ID] = tracking_run_id
    metadata[dataformat.TRACKING][dataformat.FRAME_GROUP] = frame_group_id
    return {METADATA: metadata, DATA: {DETECTIONS: detections}}
