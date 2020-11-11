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
# GNU General Public License for more objectsails.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from track.iou_tracker_qp import track_iou
from helpers.files import get_files
import json
import pathlib


config_track_default = {
    "yolo_mode": "spp",
    "sigma_l": 0.1,
    "sigma_h": 0.85,
    "sigma_iou": 0.4,
    "t_min": 12,
    "save_age": 5,
    "overwrite": True,
}


def read(objects_file):
    """
    docstring
    """
    dir = pathlib.Path(objects_file).parent
    filename = pathlib.Path(objects_file).stem.rsplit("_", 1)[0]
    # objects_suffix = pathlib.Path(objects_file).stem.rsplit("_", 1)[1]
    filetype = pathlib.Path(objects_file).suffix
    if filetype == ".json":
        with open(objects_file) as objects_file_json:
            objects = json.load(objects_file_json)
    elif filetype == ".csv":
        pass  # todo?
    else:
        raise ValueError("Filetype " + filetype + " cannot be read")
    return objects, dir, filename


def track(objects, config_track=config_track_default):
    """
    docstring
    """
    tracks_px = track_iou(
        objects,
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
    objects_files = get_files(paths, filetype)
    for objects_file in objects_files:
        objects, dir, filename = read(objects_file)
        tracks_px = track(objects)
        write(tracks_px, dir, filename, "_tracks-px", ".json")


# To Dos:
# - Perform logging by dedicated python package
