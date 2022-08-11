# OTVision: helpers for filehandling
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

import json
import shutil
from pathlib import Path
from typing import Union

from OTVision.helpers.log import log


def get_files(paths, filetypes=None, replace_filetype=False, search_subdirs=True):
    # sourcery skip: low-code-quality
    """
    Generates a list of files ending with filename based on filenames or the
    (recursive) content of folders.

    Args:
        paths ([str or list of str or Path or list of Path]): where to find
        the files.
        filetype ([str]): ending of files to find. Preceding "_" prevents adding a '.'
            If no filetype is given, filetypes of file paths given are used and
            directories are ignored. Defaults to None.
        replace_filetype ([bool]): Wheter or not to replace the filetype in file paths
            with the filetype given. Currently only applied when one filetype was given.
            Defaults to False.
        search_subdirs ([bool]): Wheter or not to search subdirs of dirs given as paths.
            Defaults to True.

    Returns:
        [list]: [list of filenames as str]
    """
    files = set()
    if type(paths) is str or isinstance(paths, Path):
        paths = [paths]
    elif type(paths) is not list:
        raise TypeError("Paths needs to be a str, a list of str, or Path object")
    if filetypes:
        if type(filetypes) is not list:
            filetypes = [filetypes]
        for idx, filetype in enumerate(filetypes):
            if type(filetype) is not str:
                raise TypeError("Filetype needs to be a str or a list of str")
            if not filetype.startswith("_"):
                if not filetype.startswith("."):
                    filetype = f".{filetype}"
                filetypes[idx] = filetype.lower()
    for path in paths:
        path = Path(path)
        if path.is_file():
            if filetypes and replace_filetype and len(filetypes) == 1 and path.suffix:
                path_with_filetype_replaced = path.with_suffix(filetypes[0])
                if path_with_filetype_replaced.is_file():
                    path = path.with_suffix(filetypes[0])
            file = str(path)
            if filetypes:
                for filetype in filetypes:
                    if path.suffix.lower() == filetype:
                        files.add(str(path))
            else:
                files.add(str(path))
        elif path.is_dir():
            for filetype in filetypes:
                for file in path.glob("**/*" if search_subdirs else "*"):
                    if file.is_file and file.suffix.lower() == filetype:
                        files.add(str(file))
        else:
            raise TypeError(
                "Paths needs to be a path as a pathlib.Path() or a str or a list of str"
            )

    return sorted(list(files))


def remove_dir(dir_to_remove: Union[str, Path]):
    dir_to_remove = Path(dir_to_remove)
    for path in dir_to_remove.glob("*"):
        if path.is_file():
            path.unlink()
        else:
            remove_dir(path)
    dir_to_remove.rmdir()


def read_json(json_file, extension=".json"):
    if isinstance(extension, str):
        extension = [extension]
    filetype = Path(json_file).suffix
    if filetype not in extension:
        raise ValueError(f"Wrong filetype {filetype}, has to be {extension}")
    try:
        with open(json_file) as f:
            dict_from_json_file = json.load(f)
    except OSError as oe:
        log.error(f"Could not open {json_file}")
        log.exception(oe)
    except json.JSONDecodeError as je:
        log.exception(
            (
                f'Unable to decode "{json_file}" as JSON.'
                f"Following exception occured: {str(je)}"
            )
        )
        log.exception(je)
    except Exception as e:
        log.exception(e)
    # BUG: "UnboundLocalError: local variable 'dict_from_json_file' referenced bef ass"
    return dict_from_json_file


def write_json(
    dict_to_write,
    file,
    extension=".json",
    overwrite=False,
):
    outfile = Path(file).with_suffix(extension)
    outfile_already_exists = outfile.is_file()
    if overwrite or not outfile_already_exists:
        with open(outfile, "w") as f:
            json.dump(dict_to_write, f, indent=4)
        if not outfile_already_exists:
            log.debug(f"{outfile} written")
        else:
            log.debug(f"{outfile} overwritten")
    else:
        log.debug(f"{outfile} already exists, not overwritten")


def denormalize(otdict, keys_width=None, keys_height=None):
    if keys_width is None:
        keys_width = ["x", "w"]
    if keys_height is None:
        keys_height = ["y", "h"]
    if otdict["det_config"]["normalized"]:
        direction = "denormalize"
        otdict = _normal_transformation(otdict, direction, keys_width, keys_height)
        otdict["det_config"]["normalized"] = False
        log.debug("Dict denormalized")
    else:
        log.debug("Dict was already denormalized")
    return otdict


def normalize(otdict, keys_width=None, keys_height=None):
    if keys_width is None:
        keys_width = ["x", "w"]
    if keys_height is None:
        keys_height = ["y", "h"]
    if not otdict["det_config"]["normalized"]:
        direction = "normalize"
        otdict = _normal_transformation(otdict, direction, keys_width, keys_height)
        otdict["det_config"]["normalized"] = True
        log.debug("Dict normalized")
    else:
        log.debug("Dict was already normalized")
    return otdict


def _normal_transformation(otdict, direction, keys_width, keys_height):
    width = otdict["vid_config"]["width"]
    height = otdict["vid_config"]["height"]
    for detection in otdict["data"]:
        for bbox in otdict["data"][detection]["classified"]:
            for key in bbox:
                if key in keys_width:
                    if direction == "normalize":
                        bbox[key] = bbox[key] / width
                    elif direction == "denormalize":
                        bbox[key] = bbox[key] * width
                elif key in keys_height:
                    if direction == "normalize":
                        bbox[key] = bbox[key] / height
                    elif direction == "denormalize":
                        bbox[key] = bbox[key] * height
    return otdict


def _get_testdatafolder():
    return str(Path(__file__).parents[2] / r"tests/data")


def is_in_format(file_path, file_formats):
    """Checks if a file path is in specified format.

    The case of a file format is ignored.

    Args:
        pathToVideo (str): the file path
        file_formats(list(str)): the file formats

    Returns:
        True if path is of format specified in file_formats.
        Otherwise False.
    """
    file = Path(file_path)
    return file.suffix.lower() in [
        file_format.lower()
        if file_format.startswith(".")
        else f".{file_format.lower()}"
        for file_format in file_formats
    ]


def unzip(file):
    file = Path(file)
    directory = file.with_suffix("")
    shutil.unpack_archive(file, directory)
    return directory
