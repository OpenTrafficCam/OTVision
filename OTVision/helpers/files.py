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

from OTVision.helpers.log import log


def get_files(
    paths: list[Path],
    filetypes: list[str] = None,
    search_subdirs: bool = True,
) -> list[Path]:
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


def remove_dir(dir_to_remove: Path):
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
