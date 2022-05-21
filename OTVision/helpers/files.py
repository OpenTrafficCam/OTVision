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
import logging
import shutil
from pathlib import Path
from typing import Union


def get_files(paths, filetypes=None, replace_filetype=False, search_subdirs=True):
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

    # Check, if paths is a str or a list
    if type(paths) is str or isinstance(paths, Path):
        paths = [paths]
    elif type(paths) is not list and not isinstance(paths, Path):
        raise TypeError("Paths needs to be a str, a list of str, or Path object")

    # Check if filetypes is str or a list and transform it
    if filetypes:
        if type(filetypes) is not list:
            filetypes = [filetypes]

        for idx, filetype in enumerate(filetypes):
            if type(filetype) is not str:
                raise TypeError("Filetype needs to be a str or a list of str")

            if not filetype.startswith("_"):
                if not filetype.startswith("."):
                    filetype = "." + filetype
                filetypes[idx] = filetype.lower()

    # add all files to a single list _files_
    for path in paths:
        path = Path(path)
        # If path is a real file add it to return list
        if path.is_file():
            # Replace filetype in path if replace_filetype is given as argument
            # and path has suffix and only one filetype was given and new path exists
            if filetypes and replace_filetype and len(filetypes) == 1 and path.suffix:
                path_with_filetype_replaced = path.with_suffix(filetypes[0])
                if path_with_filetype_replaced.is_file():
                    path = path.with_suffix(filetypes[0])
            # Add path to list of returned paths if filetype meets requirements
            file = str(path)
            if filetypes:
                for filetype in filetypes:
                    if path.suffix.lower() == filetype:
                        files.add(str(path))
            else:
                files.add(str(path))
        # If path is a real file add it to return list
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


def remove_dir(dir_path: Union[str, Path]):
    dir = Path(dir_path)
    for path in dir.glob("*"):
        if path.is_file():
            path.unlink()
        else:
            remove_dir(path)
    dir.rmdir()


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
        logging.error(
            (
                f'Could not open "{json_file}". '
                f"Following exception occured: {str(oe)}"
            )
        )
    except json.JSONDecodeError as je:
        logging.error(
            (
                f'Unable to decode "{json_file}" as JSON.'
                f"Following exception occured: {str(je)}"
            )
        )
    except Exception as e:
        logging.error(e)
        print(e)
    # BUG: "UnboundLocalError: local variable 'dict_from_json_file' referenced before assignment"
    return dict_from_json_file


def write_json(
    dict_to_write,
    file,
    extension=".json",
    overwrite=False,
):
    outfile = Path(file).with_suffix(extension)
    if overwrite or not get_files(outfile):
        with open(outfile, "w") as f:
            json.dump(dict_to_write, f, indent=4)
        logging.info("JSON written")


def denormalize(otdict, keys_width=None, keys_height=None):
    if keys_width is None:
        keys_width = ["x", "w"]
    if keys_height is None:
        keys_height = ["y", "h"]
    if otdict["det_config"]["normalized"]:
        direction = "denormalize"
        otdict = _normal_transformation(otdict, direction, keys_width, keys_height)
        otdict["det_config"]["normalized"] = False
        logging.info("Dict denormalized!")
    else:
        logging.info("Dict was not normalized!")
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
        logging.info("Dict normalized!")
    else:
        logging.info("Dict was already normalized!")
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

    if file.suffix.lower() in [
        file_format.lower()
        if file_format.startswith(".")
        else f".{file_format.lower()}"
        for file_format in file_formats
    ]:
        return True
    else:
        return False


def unzip(file):
    file = Path(file)
    directory = file.with_suffix("")
    shutil.unpack_archive(file, directory)
    return directory


if __name__ == "__main__":
    paths = "D:/tmp/"
    # paths = ["D:/tmp/tmp1", "D:\\tmp\\tmp2"]
    # paths = ["D:/tmp/tmp1/", "D:\\tmp/tmp2\\", "D:/tmp/test_objects.csv"]
    # paths = "D:/tmp/test_objects.csv"

    # filetype = "csv"
    # filetype = "CSV"
    # filetype = "_objects.csv"
    filetype = ".csv"

    files = get_files(paths, filetype)
    for file in files:
        print(file)
    print(_get_testdatafolder())
