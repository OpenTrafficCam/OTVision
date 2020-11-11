# OTVision: Python module to calculate homography matrix from reference
# points and transform trajectory points from pixel into world coordinates.

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
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from track.iou_tracker import track_iou
from track.utils import load_mot, iou
from helpers.files import get_files
import json


config_track = {
    "yolo_mode": "spp",
    "sigma_l": 0.1,
    "sigma_h": 0.85,
    "sigma_iou": 0.4,
    "t_min": 12,
    "save_age": 5,
    "overwrite": True
}


def load_det(det_file):
    """
    docstring
    """
    if input_det_file.endswith(".json"):
        pass
    elif input_det_file.endswith(".csv"):
        pass
    return det, det_file_base


def track_multi_det_files(paths):
    """
    docstring
    """
    det_files = get_files(paths, '_yolo.json')
    for det_file in det_files:
        det, det_file_base = load_det(det_file)
        trackpx = track_single_det_file(det)
        write_trackpx(trackpx, det_file_base + '_trajpx.json')


def track_single_det_file(det_file):
    """
    docstring
    """
    det = load_det(det_file)
    trackpx = track_iou(det)
    write_trackpx(trackpx, det_file + '_trajpx.json')


def write_trackpx_file(trackpx, output_track_file):
    """
    docstring
    """
    if output_track_file.endswith(".json"):
        pass
    elif output_track_file.endswith(".csv"):
        pass


# To Dos:
# - Add logging by dedicated package