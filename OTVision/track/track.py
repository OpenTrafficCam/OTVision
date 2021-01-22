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


from track.iou_tracker_qp import track_iou
from helpers.files import get_files
import json
import pathlib
from datetime import datetime


config_track_default = {
    "yolo_mode": "spp",
    "sigma_l": 0.1,
    "sigma_h": 0.85,
    "sigma_iou": 0.4,
    "t_min": 12,
    "save_age": 5,
    "overwrite": True,
}


def read(detections_file):
    """
    docstring
    """
    dir = pathlib.Path(detections_file).parent
    filename = pathlib.Path(detections_file).stem.rsplit("_", 1)[0]
    # detections_suffix = pathlib.Path(detections_file).stem.rsplit("_", 1)[1]
    filetype = pathlib.Path(detections_file).suffix
    if filetype == ".json":
        with open(detections_file) as detections_file_json:
            detections = json.load(detections_file_json)
    elif filetype == ".csv":
        pass  # todo?
    else:
        raise ValueError("Filetype " + filetype + " cannot be read")
    return detections, dir, filename


def track(detections, config_track=config_track_default):
    """
    docstring
    """
    tracks_px = track_iou(
        detections,
        config_track["sigma_l"],
        config_track["sigma_h"],
        config_track["sigma_iou"],
        config_track["t_min"],
        config_track["save_age"],
    )
    return tracks_px


def write(tracks_px, dir, filename, suffix, filetype):
    """
    docstring
    """
    if filetype == ".json":
        file = pathlib.Path(dir, filename + suffix + filetype)
        with open(file, "w") as tracks_px_file_json:
            json.dump(tracks_px, tracks_px_file_json)
    elif filetype == "pkl":
        pass  # todo?
    elif filetype == "csv":
        pass  # todo?


def main(paths, config_track=config_track_default):
    """
    docstring
    """
    filetype = "_yolo-" + config_track["yolo_mode"] + ".json"
    detections_files = get_files(paths, filetype)
    for detections_file in detections_files:
        print(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            + ": New detections file: "
            + detections_file
        )
        detections, dir, filename = read(detections_file)
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ": detections read")
        tracks_px = track(detections)
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ": detections tracked")
        write(tracks_px, dir, filename, "_tracks-px", ".json")
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ": Tracks written")


# To Dos:
# - Improve memory management (MemoryError after two 2h detections)
# - Perform logging by dedicated python package
