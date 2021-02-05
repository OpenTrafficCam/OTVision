# %%
# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Licensed under The MIT License [see LICENSE for details]
# Written by Erik Bochinski
# ---------------------------------------------------------

from track.util import iou


def make_bbox(obj):
    bbox = (obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"])
    return bbox


def center(obj):
    return (obj["xmid"], obj["ymid"])


def track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min, t_miss_max):
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
    tracks_finished = []
    vehID = 0
    vehIDs_finished = []
    # new_detections = {}

    for frame_num in detections:
        detections_frame = detections[frame_num]["classified"]
        # apply low threshold to detections
        dets = [det for det in detections_frame if det["confidence"] >= sigma_l]
        # new_detections[frame_num] = {"classified": []}
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
                    track["confidences"].append(best_match["confidence"])
                    track["labels"].append(best_match["label"])
                    track["max_confidence"] = max(
                        track["max_confidence"], best_match["confidence"]
                    )
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
                    track["age"] += 1
                    saved_tracks.append(track)
                elif (
                    track["max_confidence"] >= sigma_h 
                    and track["frames"][-1] - track["frames"][0] >= t_min
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
                    "confidences": [det["confidence"]],
                    "labels": [det["label"]],
                    "max_label": det["label"],
                    "max_confidence": det["confidence"],
                    "vehID": vehID,
                    "start_frame": int(frame_num),
                    "age": 0,
                }
            )
            """det["vehID"] = vehID
            det["first"] = True
            new_detections[frame_num]["classified"].append(det)"""
        tracks_active = updated_tracks + saved_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [
        track
        for track in tracks_active
        if (
            track["max_confidence"] >= sigma_h 
            and track["frames"][-1] - track["frames"][0] >= t_min
        )
    ]

    for track in tracks_finished:
        track["max_label"] = pd.Series(track["labels"]).mode().iat[0]

    # detections = new_detections
    """for frame_num in new_detections:
        for det in new_detections[frame_num]["classified"]:
            if det["vehID"] not in vehIDs_finished:
                det["finished"] = False
            else:
                det["finished"] = True
                # det['label'] = tracks[tracks['vehID'] == det['vehID']]['max_label']"""

    return tracks_finished
    # return new_detections, tracks_finished


# %%
