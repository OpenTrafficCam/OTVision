"""
OTVision module to track road users in frames detected by OTVision
"""
# based on IOU Tracker written by Erik Bochinski originally licensed under the
# MIT License, see
# https://github.com/bochinski/iou-tracker/blob/master/LICENSE.

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


from OTVision.config import CONFIG
from OTVision.dataformat import (
    AGE,
    BBOXES,
    CENTER,
    CLASS,
    CONFIDENCE,
    DETECTIONS,
    FINISHED,
    FIRST,
    FRAMES,
    MAX_CLASS,
    MAX_CONF,
    START_FRAME,
    TRACK_ID,
    H,
    W,
    X,
    Y,
)

from .iou_util import iou


def make_bbox(obj: dict) -> tuple[float, float, float, float]:
    """Calculates xyxy coordinates from dict of xywh.

    Args:
        obj (dict): dict of pixel values for xcenter, ycenter, width and height

    Returns:
        tuple[float, float, float, float]: xmin, ymin, xmay, ymax
    """
    return (
        obj[X] - obj[W] / 2,
        obj[Y] - obj[H] / 2,
        obj[X] + obj[W] / 2,
        obj[Y] + obj[H] / 2,
    )


def center(obj: dict) -> tuple[float, float]:
    """Retrieves center coordinates from dict.

    Args:
        obj (dict): _description_

    Returns:
        tuple[float, float]: _description_
    """
    return obj[X], obj[Y]


def track_iou(
    detections: list,  # TODO: Type hint nested list during refactoring
    sigma_l: float = CONFIG["TRACK"]["IOU"]["SIGMA_L"],
    sigma_h: float = CONFIG["TRACK"]["IOU"]["SIGMA_H"],
    sigma_iou: float = CONFIG["TRACK"]["IOU"]["SIGMA_IOU"],
    t_min: int = CONFIG["TRACK"]["IOU"]["T_MIN"],
    t_miss_max: int = CONFIG["TRACK"]["IOU"]["T_MISS_MAX"],
) -> dict:  # sourcery skip: low-code-quality
    """
    Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information
    by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.

    Args:
        detections (list): list of detections per frame, usually generated
        by util.load_mot
        sigma_l (float): low detection threshold.
        sigma_h (float): high detection threshold.
        sigma_iou (float): IOU threshold.
        t_min (float): minimum track length in frames.

    Returns:
        list: list of tracks.
    """

    _check_types(sigma_l, sigma_h, sigma_iou, t_min, t_miss_max)

    tracks_active: list = []
    # tracks_finished = []
    vehID: int = 0
    vehIDs_finished: list = []
    new_detections: dict = {}

    for frame_num in detections:
        detections_frame = detections[frame_num][DETECTIONS]
        # apply low threshold to detections
        dets = [det for det in detections_frame if det[CONFIDENCE] >= sigma_l]
        new_detections[frame_num] = {}
        updated_tracks: list = []
        saved_tracks: list = []
        for track in tracks_active:
            if dets:
                # get det with highest iou
                best_match = max(
                    dets, key=lambda x: iou(track[BBOXES][-1], make_bbox(x))
                )
                if iou(track[BBOXES][-1], make_bbox(best_match)) >= sigma_iou:
                    track[FRAMES].append(int(frame_num))
                    track[BBOXES].append(make_bbox(best_match))
                    track[CENTER].append(center(best_match))
                    track[CONFIDENCE].append(best_match[CONFIDENCE])
                    track[CLASS].append(best_match[CLASS])
                    track[MAX_CONF] = max(track[MAX_CONF], best_match[CONFIDENCE])
                    track[AGE] = 0

                    updated_tracks.append(track)

                    # remove best matching detection from detections
                    del dets[dets.index(best_match)]
                    # best_match[TRACK_ID] = track[TRACK_ID]
                    best_match[FIRST] = False
                    new_detections[frame_num][track[TRACK_ID]] = best_match

            # if track was not updated
            if not updated_tracks or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track[AGE] < t_miss_max:
                    track[AGE] += 1
                    saved_tracks.append(track)
                elif (
                    track[MAX_CONF] >= sigma_h
                    and track[FRAMES][-1] - track[FRAMES][0] >= t_min
                ):
                    # tracks_finished.append(track)
                    vehIDs_finished.append(track[TRACK_ID])
        # TODO: Alter der Tracks
        # create new tracks
        new_tracks = []
        for det in dets:
            vehID += 1
            new_tracks.append(
                {
                    FRAMES: [int(frame_num)],
                    BBOXES: [make_bbox(det)],
                    CENTER: [center(det)],
                    CONFIDENCE: [det[CONFIDENCE]],
                    CLASS: [det[CLASS]],
                    MAX_CLASS: det[CLASS],
                    MAX_CONF: det[CONFIDENCE],
                    TRACK_ID: vehID,
                    START_FRAME: int(frame_num),
                    AGE: 0,
                }
            )
            # det[TRACK_ID] = vehID
            det[FIRST] = True
            new_detections[frame_num][vehID] = det
        tracks_active = updated_tracks + saved_tracks + new_tracks

    # finish all remaining active tracks
    # tracks_finished += [
    #     track
    #     for track in tracks_active
    #     if (
    #         track["max_conf"] >= sigma_h
    #         and track["frames"][-1] - track["frames"][0] >= t_min
    #     )
    # ]

    # for track in tracks_finished:
    #     track["max_class"] = pd.Series(track["class"]).mode().iat[0]

    # TODO: #82 Use dict comprehensions in track_iou
    for frame_det in new_detections.values():
        for vehID, det in frame_det.items():
            det[FINISHED] = vehID in vehIDs_finished
            det[TRACK_ID] = vehID
    # return tracks_finished
    # TODO: #83 Remove unnessecary code (e.g. for tracks_finished) from track_iou
    return new_detections


def _check_types(
    sigma_l: float, sigma_h: float, sigma_iou: float, t_min: int, t_miss_max: int
) -> None:
    """Raise ValueErrors if wrong types"""

    if not isinstance(sigma_l, (int, float)):
        raise ValueError("sigma_l has to be int or float")
    if not isinstance(sigma_h, (int, float)):
        raise ValueError("sigma_h has to be int or float")
    if not isinstance(sigma_iou, (int, float)):
        raise ValueError("sigma_iou has to be int or float")
    if not isinstance(t_min, int):
        raise ValueError("t_min has to be int")
    if not isinstance(t_miss_max, int):
        raise ValueError("t_miss_max has to be int")
