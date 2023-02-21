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

import bz2
import shutil
from pathlib import Path
from typing import Union

import ujson

from OTVision.config import CONFIG
from OTVision.dataformat import INPUT_FILE_PATH
from OTVision.helpers.log import log

ENCODING = "UTF-8"
COMPRESSED_FILETYPE = ".bz2"


def get_files(
    paths: list[Path],
    filetypes: Union[list[str], None] = None,
    search_subdirs: bool = True,
) -> list[Path]:
    """
    Generates a list of files ending with filename based on filenames or the
    (recursive) content of folders.

    Args:
        paths (list[Path]): where to find the files.
        filetype (list[str]): ending of files to find. Preceding "_" prevents adding a
        '.'
            If no filetype is given, filetypes of file paths given are used and
            directories are ignored. Defaults to None.
        search_subdirs (bool): Wheter or not to search subdirs of dirs given as paths.
            Defaults to True.

    Raises:
        TypeError: If type of paths is not list
        TypeError: If type of path in paths is not Path or subclass
        TypeError: If type of filetypes is not list
        TypeError: If type of filetype in filetypes is not str
        TypeError: If path in paths is neither valid file nor dir

    Returns:
        list[Path]: List of files
    """
    files = set()
    if type(paths) is not list:
        raise TypeError("Paths needs to be a list of pathlib.Path")
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
        if not isinstance(path, Path):
            raise TypeError("Paths needs to be a list of pathlib.Path")
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
                        if file.is_file() and file.suffix.lower() == filetype:
                            files.add(file)
        else:
            raise TypeError("Paths needs to be a list of pathlib.Path")

    return sorted(list(files))


def replace_filetype(
    files: list[Path], new_filetype: str, old_filetype: Union[str, None] = None
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
        raise TypeError("Paths needs to be a list of pathlib.Path")
    new_paths = []
    for path in files:
        if not isinstance(path, Path):
            raise TypeError("Paths needs to be a list of pathlib.Path")
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


def _remove_dir(dir_to_remove: Path) -> None:
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


def read_json(
    json_file: Path,
    filetype: str = ".json",
    decompress: bool = True,
) -> dict:
    """Read a json file of a specific filetype to a dict.

    Args:
        json_file (Path): json file to read
        filetype (str, optional): filetype to check json file against.
            Defaults to ".json".
        decompress: (bool, optional): decompress output with bzip2.
            If `filetype` is not `.bz2`, decompress will be set to False.
            Defaults to True.

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
        if decompress:
            with bz2.open(json_file, "rt", encoding=ENCODING) as input:
                dict_from_json_file = ujson.load(input)
        else:
            with open(json_file, "r", encoding=ENCODING) as input:
                dict_from_json_file = ujson.load(input)
        log.info(f"{json_file} read")
        return dict_from_json_file
    except OSError:
        log.exception(f"Could not open {json_file}")
        raise
    except ujson.JSONDecodeError:
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
    compress: bool = True,
) -> None:
    """Write a json file from a dict to a specific filetype.

    Args:
        dict_to_write (dict): dict to write
        file (Path): file path. Can have other filetype, which will be overwritten.
        filetype (str, optional): filetype of file to be written.
            Defaults to ".json".
        overwrite (bool, optional): Whether or not to overwrite an existing file.
            Defaults to False.
        compress: (bool, optional): compress input with bzip2.
            If `filetype` is not `.bz2`, compress will be set to False.
            Defaults to True.
    """
    outfile = Path(file).with_suffix(filetype)
    outfile_already_exists = outfile.is_file()
    if overwrite or not outfile_already_exists:
        if compress:
            with bz2.open(outfile, "wt", encoding=ENCODING) as output:
                ujson.dump(dict_to_write, output)
        else:
            with open(outfile, "w", encoding=ENCODING) as output:
                ujson.dump(dict_to_write, output)
        if not outfile_already_exists:
            log.debug(f"{outfile} written")
        else:
            log.debug(f"{outfile} overwritten")
    else:
        log.debug(f"{outfile} already exists, not overwritten. Set overwrite=True")


# TODO: Type hint nested dict during refactoring
def _check_and_update_metadata_inplace(otdict: dict) -> None:
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


# TODO: Type hint nested dict during refactoring
def denormalize_bbox(
    otdict: dict,
    keys_width: Union[list[str], None] = None,
    keys_height: Union[list[str], None] = None,
    metadata: dict[str, dict] = {},
) -> dict:
    """Denormalize all bbox references in detections or tracks dict from percent to px.

    Args:
        otdict (dict): dict of detections or tracks
        keys_width (list[str], optional): list of keys describing horizontal position.
            Defaults to ["x", "w"].
        keys_height (list[str], optional): list of keys describing vertical position.
            Defaults to ["y", "h"].
        metadata (dict[str, dict]): dict of metadata per input file.

    Returns:
        _type_: Denormalized dict.
    """
    if keys_width is None:
        keys_width = ["x", "w"]
    if keys_height is None:
        keys_height = ["y", "h"]
    log.debug("Denormalize frame wise")
    otdict = _denormalize_transformation(otdict, keys_width, keys_height, metadata)
    return otdict


# TODO: Type hint nested dict during refactoring
def _denormalize_transformation(
    otdict: dict,
    keys_width: list[str],
    keys_height: list[str],
    metadata: dict[str, dict] = {},
) -> dict:
    """Helper to do the actual denormalization.

    Args:
        otdict (dict): dict of detections or tracks
        keys_width (list[str]): list of keys describing horizontal position.
            Defaults to ["x", "w"].
        keys_height (list[str]): list of keys describing vertical position.
            Defaults to ["y", "h"].
        metadata (dict[str, dict]): dict of metadata per input file.

    Returns:
        dict: denormalized dict
    """
    changed_files = set()

    for frame in otdict["data"].values():
        input_file = frame[INPUT_FILE_PATH]
        metadate = metadata[input_file]
        width = metadate["vid"]["width"]
        height = metadate["vid"]["height"]
        is_normalized = metadate["det"]["normalized"]
        if is_normalized:
            changed_files.add(input_file)
            for bbox in frame["classified"]:
                for key in bbox:
                    if key in keys_width:
                        bbox[key] = bbox[key] * width
                    elif key in keys_height:
                        bbox[key] = bbox[key] * height

    for file in changed_files:
        metadata[file]["det"]["normalized"] = False
    return otdict


# TODO: Type hint nested dict during refactoring
def normalize_bbox(
    otdict: dict,
    keys_width: Union[list[str], None] = None,
    keys_height: Union[list[str], None] = None,
    metadata: dict[str, dict] = {},
) -> dict:
    """Normalize all bbox references in detections or tracks dict from percent to px.

    Args:
        otdict (dict): dict of detections or tracks
        keys_width (list[str], optional): list of keys describing horizontal position.
            Defaults to ["x", "w"].
        keys_height (list[str], optional): list of keys describing vertical position.
            Defaults to ["y", "h"].
        metadata (dict[str, dict]): dict of metadata per input file.

    Returns:
        _type_: Normalized dict.
    """
    if keys_width is None:
        keys_width = ["x", "w"]
    if keys_height is None:
        keys_height = ["y", "h"]
    if not otdict["metadata"]["normalized"]:
        otdict = _normalize_transformation(
            otdict,
            keys_width,
            keys_height,
            metadata,
        )
        log.debug("Dict normalized")
    else:
        log.debug("Dict was already normalized")
    return otdict


# TODO: Type hint nested dict during refactoring
def _normalize_transformation(
    otdict: dict,
    keys_width: list[str],
    keys_height: list[str],
    metadata: dict[str, dict] = {},
) -> dict:
    """Helper to do the actual normalization.

    Args:
        otdict (dict): dict of detections or tracks
        keys_width (list[str]): list of keys describing horizontal position.
            Defaults to ["x", "w"].
        keys_height (list[str]): list of keys describing vertical position.
            Defaults to ["y", "h"].


    Returns:
        dict: Normalized dict
    """
    changed_files = set()

    for frame in otdict["data"].values():
        input_file = frame[INPUT_FILE_PATH]
        metadate = metadata[input_file]
        width = metadate["vid"]["width"]
        height = metadate["vid"]["height"]
        is_denormalized = not metadate["normalized"]
        if is_denormalized:
            changed_files.add(input_file)
            for bbox in frame["classified"]:
                for key in bbox:
                    if key in keys_width:
                        bbox[key] = bbox[key] / width
                    elif key in keys_height:
                        bbox[key] = bbox[key] / height

    for file in changed_files:
        metadata[file]["normalized"] = True
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


def is_video(file: Path) -> bool:
    """Checks if a file is a video according to its filetype

    Args:
        file (Path): file to check

    Returns:
        bool: whether or not the file is a video
    """
    return file.suffix.lower() in CONFIG["FILETYPES"]["VID"]


def is_image(file: Path) -> bool:
    """Checks if a file is an image according to its filetype

    Args:
        file (Path): file to check

    Returns:
        bool: whether or not the file is an image
    """
    return file.suffix.lower() in CONFIG["FILETYPES"]["IMG"]


def unzip(file: Path) -> Path:
    """Unpack a zip archive to a directory of same name.

    Args:
        file (Path): zip to unpack

    Returns:
        Path: unzipped directory
    """
    directory = file.with_suffix("")
    shutil.unpack_archive(file, directory)
    return directory
