import json
from copy import deepcopy

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files, read_json, write_json


def main(tracks_files, reftpts_file):
    reftpts_file = read_refpts(reftpts_file=reftpts_file)
    tracks_files = get_files(tracks_files, filetypes=".ottrk")
    print(tracks_files)
    for tracks_file in tracks_files:
        tracks = read_tracks(tracks_file)
        already_utm = "utm" in tracks["trk_config"] and tracks["trk_config"]["utm"]
        if CONFIG["TRANSFORM"]["OVERWRITE"] or not already_utm:
            tracks_utm = transform(tracks)
            write_tracks(tracks_utm, tracks_file)


def read_tracks(tracks_file):
    return read_json(tracks_file, extension=CONFIG["FILETYPES"]["TRACK"])


def read_refpts(reftpts_file):
    return read_json(reftpts_file, extension=CONFIG["FILETYPES"]["REFPTS"])


def transform(tracks):
    tracks_utm = deepcopy(tracks)
    tracks_utm["trk_config"]["utm"] = False
    # TODO: Calc homography
    # TODO: Transform
    return tracks_utm


def write_refpts(refpts, refpts_file):
    write_json(
        dict_to_write=refpts,
        file=refpts_file,
        extension=CONFIG["FILETYPES"]["REFPTS"],
        overwrite=True,
    )


def write_tracks(tracks_utm, tracks_file):
    write_json(
        dict_to_write=tracks_utm,
        file=tracks_file,
        extension=".ottrk",
        overwrite=True,
    )
