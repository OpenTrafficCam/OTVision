# %%
# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Licensed under The MIT License [see LICENSE for details]
# Written by Erik Bochinski
# ---------------------------------------------------------

from track.iou_util import iou


def make_bbox(obj):
    bbox = (
        obj["x"] - obj["w"] / 2,
        obj["y"] - obj["h"] / 2,
        obj["x"] + obj["w"] / 2,
        obj["y"] + obj["h"] / 2,
    )
    return bbox


def center(obj):
    center = (obj["x"], obj["y"])

    return center


def track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min, t_miss_max):
    """
    Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information
    by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.

    Args:
         detections (dict): dict of detections in % of video size
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.

    Returns:
        list: list of tracks in % of video size.
    """

    tracks_active = []
    tracks_finished = []
    vehID = 0
    vehIDs_finished = []
    new_detections = {}

    for frame_num in detections:
        detections_frame = detections[frame_num]["classified"]
        # apply low threshold to detections
        dets = [det for det in detections_frame if det["conf"] >= sigma_l]
        new_detections[frame_num] = {"classified": []}
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
                    track["confs"].append(best_match["conf"])
                    track["classes"].append(best_match["class"])
                    track["max_conf"] = max(track["max_conf"], best_match["conf"])
                    track["age"] = 0

                    updated_tracks.append(track)

                    # remove best matching detection from detections
                    del dets[dets.index(best_match)]
                    """best_match["vehID"] = track["vehID"]
                    best_match["first"] = False
                    new_detections[frame_num]["classified"].append(best_match)"""

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track["age"] < t_miss_max:
                    track["frames"].append(track["frames"][-1])
                    track["bboxes"].append(track["bboxes"][-1])
                    track["center"].append(track["center"][-1])
                    track["confs"].append(track["confs"][-1])
                    track["classes"].append(track["classes"][-1])
                    track["age"] += 1
                    saved_tracks.append(track)
                elif track["max_conf"] >= sigma_h and len(track["frames"]) >= (
                    t_min + track["age"]
                ):
                    tracks_finished.append(track)
                    vehIDs_finished.append(track["vehID"])
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
                    "confs": [det["conf"]],
                    "classes": [det["class"]],
                    "max_class": det["class"],
                    "max_conf": det["conf"],
                    "vehID": vehID,
                    "start_frame": int(frame_num),
                    "age": 0,
                }
            )
            # det["vehID"] = vehID
            # det["first"] = True
            # new_detections[frame_num]["classified"].append(det)
        tracks_active = updated_tracks + saved_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [
        track
        for track in tracks_active
        if track["max_conf"] >= sigma_h and len(track["bboxes"]) >= t_min
    ]

    for track in tracks_finished:
        track["max_class"] = max(track["classes"], key=track["classes"].count)

    # detections = new_detections
    for frame_num in new_detections:
        for det in new_detections[frame_num]["classified"]:
            if det["vehID"] not in vehIDs_finished:
                det["finished"] = False
            else:
                det["finished"] = True
                # det['class'] = tracks[tracks['vehID'] == det['vehID']]['max_class']

    return new_detections, tracks_finished
