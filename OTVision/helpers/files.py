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

from pathlib import Path


def get_files(paths, filetype):
    """
    Generates a list of files ending with filename based on filenames or the recursive
    contend of folders.

    Args:
        paths ([str or list of str]): where to find the files
        filetype ([str]): ending of the files to find. Preceding "_" prevents adding a '.'

    Returns:
        [list]: [list of filenames as str]
    """

    files = set()
    pathlist = []

    # check, if _paths_ is a list or str
    if type(paths) is list:
        pathlist.extend(paths)
    elif type(paths) is str:
        pathlist.append(paths)
    else:
        raise TypeError("Paths needs to be str or list of str")

    # check if _filename_ is str and transform it
    if type(filetype) is str:
        if not filetype.startswith("_"):
            if not filetype.startswith("."):
                filetype = "." + filetype
            filetype = filetype.lower()
    else:
        raise TypeError("filetype needs to be a str")

    # add all files to a single list _files_
    for path in pathlist:
        path = Path(path)
        if path.is_file():
            file = str(path)
            if file.endswith(filetype):
                files.add(file)
        elif path.is_dir():
            for file in path.glob("**/*" + filetype):
                file = str(file)
                files.add(file)

    return sorted(list(files))

  
def remove_dir(dir: str):
    dir = Path(dir)
    for path in dir.glob("*"):
        if path.is_file():
            path.unlink()
        else:
            remove_dir(path)
    dir.rmdir()


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
