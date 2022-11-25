"""
OTVision helpers for filehandling
"""
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

import json
import shutil
from pathlib import Path

from OTVision.config import CONFIG
from OTVision.helpers.log import log


def get_files(
    paths: list[Path],
    filetypes: list[str] = None,
    search_subdirs: bool = True,
) -> list[Path]:
    """
    Generates a list of files ending with filename based on filenames or the
    (recursive) content of folders.

    Args:
        paths ([str or list of str or Path or list of Path]): where to find
        the files.
        filetype ([str]): ending of files to find. Preceding "_" prevents adding a '.'
            If no filetype is given, filetypes of file paths given are used and
            directories are ignored. Defaults to None.
        search_subdirs ([bool]): Wheter or not to search subdirs of dirs given as paths.
            Defaults to True.

    Returns:
        [list]: [list of filenames as str]
    """
    files = set()
    if type(paths) is not list:
        raise TypeError("Paths needs to a list of pathlib.Path")
    if filetypes:
        if type(filetypes) is not list:
            raise TypeError("Filetypes needs to be a list of str")
        for idx, filetype in enumerate(filetypes):
            if type(filetype) is not str:
                raise TypeError("Filetypes needs to be a list of str")
            if not filetype.startswith("_"):
                if not filetype.startswith("."):
                    filetype = f".{filetype}"
                filetypes[idx] = filetype.lower()
    for path in paths:
        path = Path(path)
        if path.is_file():
            file = path
            if filetypes:
                for filetype in filetypes:
                    if path.suffix.lower() == filetype:
                        files.add(path)
            else:
                files.add(path)
        elif path.is_dir():
            if filetypes:
                for filetype in filetypes:
                    for file in path.glob("**/*" if search_subdirs else "*"):
                        if file.is_file and file.suffix.lower() == filetype:
                            files.add(file)
        else:
            raise TypeError("Paths needs to be a list of pathlib.Path")

    return sorted(list(files))


def replace_filetype(
    files: list[Path], new_filetype: str, old_filetype: str = None
) -> list[Path]:
    """In a list of files, replace the filetype of all files of a certain old_filetype
    by a new_filetype. If no old_filetype is given, replace tha filetype of all files.
    Directories remain unchanged in the new list.

    Args:
        paths (list[Path]): List of paths (can be files or directories).
        new_filetype (str): New file type after replacement.
        old_filetype (str): File type to be replaced. If None, filetypes of all files
            will be replaced.
            Defaults to None.

    Raises:
        TypeError: If files is not a list of pathlib.Path
        TypeError: If one of the files is not a file (but, for example, a dir)

    Returns:
        list[Path]: List of paths with file type replaced
    """

    if type(files) is not list:
        raise TypeError("Paths needs to a list of pathlib.Path")
    new_paths = []
    for path in files:
        if type(path) is not Path:
            raise TypeError("Paths needs to a list of pathlib.Path")
        if path.is_file():
            if old_filetype and path.suffix.lower() != old_filetype.lower():
                continue
            new_path = path.with_suffix(new_filetype)
            new_paths.append(new_path)
        elif path.is_dir():
            raise TypeError("files has to be a list of files without dirs")
        else:
            raise TypeError("files has to be a list of existing files")

    return new_paths


def _remove_dir(dir_to_remove: Path):
    """Helper to remove a directory and all of its subdirectories.

    Args:
        dir_to_remove (Path): directory to remove
    """
    for path in dir_to_remove.glob("*"):
        if path.is_file():
            path.unlink()
        else:
            _remove_dir(path)
    dir_to_remove.rmdir()


def read_json(json_file: Path, filetype: str = ".json") -> dict:
    """Read a json file of a specific filetype to a dict.

    Args:
        json_file (Path): json file to read
        filetype (str, optional): filetype to check json file against.
            Defaults to ".json".

    Raises:
        TypeError: If file is not pathlib.Path
        ValueError: If file is not of filetype given

    Returns:
        dict: dict read from json file
    """
    if not isinstance(json_file, Path):
        raise TypeError("json_file has to be of type pathlib.Path")
    filetype = json_file.suffix
    if json_file.suffix != filetype:
        raise ValueError(f"Wrong filetype {str(json_file)}, has to be {filetype}")
    try:
        with open(json_file) as f:
            dict_from_json_file = json.load(f)
        log.info(f"{json_file} read")
        return dict_from_json_file
    except OSError:
        log.exception(f"Could not open {json_file}")
        raise
    except json.JSONDecodeError:
        log.exception(f'Unable to decode "{json_file}" as JSON.')
        raise
    except Exception:
        log.exception("")
        raise


def write_json(
    dict_to_write: dict,
    file: Path,
    filetype: str = ".json",
    overwrite: bool = False,
):
    """Write a json file from a dict to a specific filetype.

    Args:
        dict_to_write (dict): dict to write
        file (Path): file path. Can have other filetype, which will be overwritten.
        filetype (str, optional): filetype of file to be written.
            Defaults to ".json".
        overwrite (bool, optional): Whether or not to overwrite an existing file.
            Defaults to False.
    """
    outfile = Path(file).with_suffix(filetype)
    outfile_already_exists = outfile.is_file()
    if overwrite or not outfile_already_exists:
        with open(outfile, "w") as f:
            json.dump(dict_to_write, f, indent=4)
        if not outfile_already_exists:
            log.debug(f"{outfile} written")
        else:
            log.debug(f"{outfile} overwritten")
    else:
        log.debug(f"{outfile} already exists, not overwritten. Set overwrite=True")


def _check_and_update_metadata_inplace(otdict: dict):
    """Check if dict of detections or tracks has subdict metadata.
        If not, try to convert from historic format.
        Atttention: Updates the input dict inplace.

    Args:
        otdict (dict): dict of detections or tracks
    """
    if "metadata" in otdict:
        return
    try:
        otdict["metadata"] = {}
        if "vid_config" in otdict:
            otdict["metadata"]["vid"] = otdict["vid_config"]
        if "det_config" in otdict:
            otdict["metadata"]["det"] = otdict["det_config"]
        if "trk_config" in otdict:
            otdict["metadata"]["trk"] = otdict["trk_config"]
        log.info("metadata updated from historic format to new format")
    except Exception:
        log.exception("metadata not found and not in historic config format")
        raise


def denormalize_bbox(
    otdict: dict, keys_width: list[str] = None, keys_height: list[str] = None
):
    """Denormalize all bbox references in detections or tracks dict from percent to px.

    Args:
        otdict (dict): dict of detections or tracks
        keys_width (list[str], optional): list of keys describing horizontal position.
            Defaults to ["x", "w"].
        keys_height (list[str], optional): list of keys describing vertical position.
            Defaults to ["y", "h"].

    Returns:
        _type_: Denormalized dict.
    """
    if keys_width is None:
        keys_width = ["x", "w"]
    if keys_height is None:
        keys_height = ["y", "h"]
    if otdict["metadata"]["det"]["normalized"]:
        direction = "denormalize"
        otdict = _normal_transformation(otdict, direction, keys_width, keys_height)
        otdict["metadata"]["det"]["normalized"] = False
        log.debug("Dict denormalized")
    else:
        log.debug("Dict was already denormalized")
    return otdict


def normalize_bbox(
    otdict: dict, keys_width: list[str] = None, keys_height: list[str] = None
):
    """Normalize all bbox references in detections or tracks dict from percent to px.

    Args:
        otdict (dict): dict of detections or tracks
        keys_width (list[str], optional): list of keys describing horizontal position.
            Defaults to ["x", "w"].
        keys_height (list[str], optional): list of keys describing vertical position.
            Defaults to ["y", "h"].

    Returns:
        _type_: Normalized dict.
    """
    if keys_width is None:
        keys_width = ["x", "w"]
    if keys_height is None:
        keys_height = ["y", "h"]
    if not otdict["metadata"]["normalized"]:
        direction = "normalize"
        otdict = _normal_transformation(otdict, direction, keys_width, keys_height)
        otdict["metadata"]["normalized"] = True
        log.debug("Dict normalized")
    else:
        log.debug("Dict was already normalized")
    return otdict


def _normal_transformation(
    otdict: dict, direction: str, keys_width: list[str], keys_height: list[str]
) -> dict:
    """Helper to do the actual normalization or denormalization.
        (Reduces duplicate code snippets)

    Args:
        otdict (dict): dict of detections or tracks
        direction (str): "normalize" or "denormalize"
        keys_width (list[str]): list of keys describing horizontal position.
            Defaults to ["x", "w"].
        keys_height (list[str]): list of keys describing vertical position.
            Defaults to ["y", "h"].

    Returns:
        dict: Normalized or denormalized dict
    """
    width = otdict["metadata"]["vid"]["width"]
    height = otdict["metadata"]["vid"]["height"]
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


def has_filetype(file: Path, filetypes: list[str]) -> bool:
    """Checks if a file has a specified filetype.

    The case of a filetype is ignored.

    Args:
        file (Path): The path to the file
        file_formats(list(str)): The valid filetypes

    Returns:
        True if file is of filetype specified in filetypes.
        Otherwise False.
    """

    return file.suffix.lower() in [
        filetype.lower() if filetype.startswith(".") else f".{filetype.lower()}"
        for filetype in filetypes
    ]


def unzip(file):
    file = Path(file)
    directory = file.with_suffix("")
    shutil.unpack_archive(file, directory)
    return directory
