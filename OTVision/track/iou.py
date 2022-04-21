# OTVision: Python module to track road users in frames detected by OTVision
# based on IOU Tracker written by Erik Bochinski originally licensed under the
# MIT License, see <https://github.com/bochinski/iou-tracker>.

# Copyright (C) 2021 OpenTrafficCam Contributors
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


import pandas as pd

from OTVision.config import CONFIG

from .iou_util import iou


def make_bbox(obj):
    return (
        obj["x"] - obj["w"] / 2,
        obj["y"] - obj["h"] / 2,
        obj["x"] + obj["w"] / 2,
        obj["y"] + obj["h"] / 2,
    )


def center(obj):
    return obj["x"], obj["y"]


def track_iou(
    detections,
    sigma_l=CONFIG["TRACK"]["IOU"]["SIGMA_L"],
    sigma_h=CONFIG["TRACK"]["IOU"]["SIGMA_H"],
    sigma_iou=CONFIG["TRACK"]["IOU"]["SIGMA_IOU"],
    t_min=CONFIG["TRACK"]["IOU"]["T_MIN"],
    t_miss_max=CONFIG["TRACK"]["IOU"]["T_MISS_MAX"],
):
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

    tracks_active = []
    # tracks_finished = []
    tracks_geojson = {"type": "FeatureCollection", "features": []}
    vehID = 0
    vehIDs_finished = []
    new_detections = {}

    for frame_num in detections:
        detections_frame = detections[frame_num]["classified"]
        # apply low threshold to detections
        dets = [det for det in detections_frame if det["conf"] >= sigma_l]
        new_detections[frame_num] = {}
        updated_tracks = []
        saved_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(
                    dets, key=lambda x: iou(track["bboxes"][-1], make_bbox(x))
                )
                if iou(track["bboxes"][-1], make_bbox(best_match)) >= sigma_iou:
                    track["frames"].append(int(frame_num))
                    track["bboxes"].append(make_bbox(best_match))
                    track["center"].append(center(best_match))
                    track["conf"].append(best_match["conf"])
                    track["class"].append(best_match["class"])
                    track["max_conf"] = max(track["max_conf"], best_match["conf"])
                    track["age"] = 0

                    updated_tracks.append(track)

                    # remove best matching detection from detections
                    del dets[dets.index(best_match)]
                    # best_match["vehID"] = track["vehID"]
                    best_match["first"] = False
                    new_detections[frame_num][track["vehID"]] = best_match

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track["age"] < t_miss_max:
                    track["age"] += 1
                    saved_tracks.append(track)
                elif (
                    track["max_conf"] >= sigma_h
                    and track["frames"][-1] - track["frames"][0] >= t_min
                ):
                    # tracks_finished.append(track)
                    vehIDs_finished.append(track["vehID"])
                    tracks_geojson["features"].append(
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "LineString",
                                "coordinates": track["center"],
                            },
                            "properties": {
                                "max_conf": track["max_conf"],
                                "ID": track["vehID"],
                                "start_frame": track["frames"][0],
                            },
                        }
                    )
        # TODO: Alter der Tracks
        # create new tracks
        new_tracks = []
        for det in dets:
            vehID += 1
            new_tracks.append(
                {
                    "frames": [int(frame_num)],
                    "bboxes": [make_bbox(det)],
                    "center": [center(det)],
                    "conf": [det["conf"]],
                    "class": [det["class"]],
                    "max_class": det["class"],
                    "max_conf": det["conf"],
                    "vehID": vehID,
                    "start_frame": int(frame_num),
                    "age": 0,
                }
            )
            # det["vehID"] = vehID
            det["first"] = True
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
    tracks_geojson["features"] += [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": track["center"],
            },
            "properties": {
                "max_conf": track["max_conf"],
                "ID": track["vehID"],
                "start_frame": track["frames"][0],
            },
        }
        for track in tracks_active
        if (
            track["max_conf"] >= sigma_h
            and track["frames"][-1] - track["frames"][0] >= t_min
        )
    ]

    # for track in tracks_finished:
    #     track["max_class"] = pd.Series(track["class"]).mode().iat[0]
    for track_geojson in tracks_geojson["features"]:
        track_geojson["properties"]["max_class"] = (
            pd.Series(track["class"]).mode().iat[0]
        )
    detections = new_detections
    # TODO: #82 Use dict comprehensions in track_iou
    for frame_num, frame_det in new_detections.items():
        for vehID, det in frame_det.items():
            if vehID not in vehIDs_finished:
                det["finished"] = False
            else:
                det["finished"] = True
                # det["label"] = tracks[tracks["vehID"] == det["vehID"]]["max_label"]

    # return tracks_finished
    # TODO: #83 Remove unnessecary code (e.g. for tracks_finished) from track_iou
    return (
        new_detections,
        tracks_geojson,
        vehIDs_finished,
    )
