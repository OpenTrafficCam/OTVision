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
import logging
import shutil
import time
from pathlib import Path
from typing import Iterable, Union

import ijson
import ujson

from OTVision import dataformat
from OTVision.config import CONFIG
from OTVision.dataformat import INPUT_FILE_PATH, METADATA
from OTVision.helpers.log import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)

ENCODING = "UTF-8"
COMPRESSED_FILETYPE = ".bz2"


START_DATE = "start_date"
FILE_NAME_PATTERN = r".*(?P<start_date>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}).*"
HOSTNAME = "hostname"
FULL_FILE_NAME_PATTERN = (
    r"(?P<hostname>.*?)_(?P<start_date>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})*.*"
)


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
        files (list[Path]): List of paths (can be files or directories).
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


def check_if_all_paths_exist(paths: list[Path]) -> None:
    for path in paths:
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(f"{path} is not an existing file or directory")


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


def read_json_bz2_event_stream(path: Path) -> Iterable[tuple[str, str, str]]:
    """
    Provide lazy data stream reading the bzip2 compressed file
    at the given path and interpreting it as json objects.
    """
    # TODO error handling
    stream = bz2.BZ2File(path)
    return ijson.parse(stream)


def metadata_from_json_events(parse_events: Iterable[tuple[str, str, str]]) -> dict:
    """
    Extract the metadata block of the ottrk data format
    from the given json parser event stream.
    """
    result: dict
    for data in ijson.items(parse_events, METADATA):
        result = data
        break
    return result


def read_json_bz2_metadata(path: Path) -> dict:
    try:
        return metadata_from_json_events(read_json_bz2_event_stream(path))
    except EOFError as cause:
        log.exception(f'Unable to read "{path}" as JSON.', exc_info=cause)
        raise cause


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
        t_json_start = time.perf_counter()
        if decompress:
            log.debug(f"Read and decompress {json_file}")
            with bz2.open(json_file, "rt", encoding=ENCODING) as input:
                dict_from_json_file = ujson.load(input)
        else:
            log.debug(f"Read {json_file} withoud decompression")
            with open(json_file, "r", encoding=ENCODING) as input:
                dict_from_json_file = ujson.load(input)
        log.debug(f"Succesfully read {json_file}")
        t_json_end = time.perf_counter()
        log.debug(f"Reading {json_file} took: {t_json_end - t_json_start:0.4f}s")
        return dict_from_json_file
    except OSError as cause:
        log.exception(f"Could not open {json_file}", exc_info=cause)
        raise cause
    except ujson.JSONDecodeError as cause:
        log.exception(f'Unable to decode "{json_file}" as JSON.', exc_info=cause)
        raise cause
    except Exception as cause:
        log.exception("", exc_info=cause)
        raise cause


# TODO: Type hint nested dict during refactoring
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
        t_json_start = time.perf_counter()
        if compress:
            log.debug(f"Compress and write {outfile}")
            with bz2.open(outfile, "wt", encoding=ENCODING) as output:
                ujson.dump(dict_to_write, output)
        else:
            log.debug(f"Write {outfile} without compression")
            with open(outfile, "w", encoding=ENCODING) as output:
                ujson.dump(dict_to_write, output)
        t_json_end = time.perf_counter()

        if not outfile_already_exists:
            log.debug(f"Successfully wrote {outfile}")
        else:
            log.debug(f"Successfully overwrote {outfile}")

        log.debug(f"Writing {outfile} took: {t_json_end - t_json_start:0.4f}s")
    else:
        log.debug(f"{outfile} already exists, not overwritten. Set overwrite=True")


# TODO: Type hint nested dict during refactoring
def get_metadata(otdict: dict) -> dict:
    """Check if dict of detections or tracks has subdict metadata.
    If not, try to convert from historic format.

    Args:
        otdict (dict): dict of detections or tracks
    """
    if dataformat.METADATA in otdict:
        return otdict[dataformat.METADATA]
    try:
        metadata = {}
        if "vid_config" in otdict:
            metadata[dataformat.VIDEO] = otdict["vid_config"]
        if "det_config" in otdict:
            metadata[dataformat.DETECTION] = otdict["det_config"]
        if "trk_config" in otdict:
            metadata[dataformat.TRACKING] = otdict["trk_config"]
        log.info("new metadata created from historic information")
        return metadata

    except Exception:
        log.exception("Metadata not found and not in historic config format")
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
        keys_width = [dataformat.X, dataformat.W]
    if keys_height is None:
        keys_height = [dataformat.Y, dataformat.H]
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

    for frame in otdict[dataformat.DATA].values():
        input_file = frame[INPUT_FILE_PATH]
        metadate = metadata[input_file]
        width = metadate[dataformat.VIDEO][dataformat.WIDTH]
        height = metadate[dataformat.VIDEO][dataformat.HEIGHT]
        is_normalized = metadate[dataformat.DETECTION][dataformat.NORMALIZED_BBOX]
        if is_normalized:
            changed_files.add(input_file)
            for bbox in frame[dataformat.DETECTIONS]:
                for key in bbox:
                    if key in keys_width:
                        bbox[key] = bbox[key] * width
                    elif key in keys_height:
                        bbox[key] = bbox[key] * height

    for file in changed_files:
        metadata[file][dataformat.DETECTION][dataformat.NORMALIZED_BBOX] = False
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
        keys_width = [dataformat.X, dataformat.W]
    if keys_height is None:
        keys_height = [dataformat.Y, dataformat.H]
    if not otdict[dataformat.METADATA][dataformat.NORMALIZED_BBOX]:
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

    for frame in otdict[dataformat.DATA].values():
        input_file = frame[INPUT_FILE_PATH]
        metadate = metadata[input_file]
        width = metadate[dataformat.VIDEO][dataformat.WIDTH]
        height = metadate[dataformat.VIDEO][dataformat.HEIGHT]
        is_denormalized = not metadate[dataformat.NORMALIZED_BBOX]
        if is_denormalized:
            changed_files.add(input_file)
            for bbox in frame[dataformat.DETECTIONS]:
                for key in bbox:
                    if key in keys_width:
                        bbox[key] = bbox[key] / width
                    elif key in keys_height:
                        bbox[key] = bbox[key] / height

    for file in changed_files:
        metadata[file][dataformat.NORMALIZED_BBOX] = True
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


class InproperFormattedFilename(Exception):
    pass
