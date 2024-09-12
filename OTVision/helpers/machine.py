"""
OTVision helpers to gather information about the machine and the system
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


import platform

OS = platform.system().replace("Darwin", "Mac")
"""OS OTVision is currently running on"""

ON_WINDOWS = OS == "Windows"
"""Wether OS is Windows or not"""

ON_LINUX = OS == "Linux"
"""Wether OS is Linux or not"""

ON_MAC = OS == "Mac"
"""Wether OS is MacOS or not"""

OS_RELEASE = platform.release()
"""Release of the OS OTVision is currently running on"""

OS_VERSION = platform.version()
"""Specific version of the OS OTVision is currently running on"""

PY_MAJOR_VERSION = int(platform.python_version_tuple()[0])
"""Python major version digit (e.g. 3 for 3.9.5) OTVision is currently running with"""

PY_MINOR_VERSION = int(platform.python_version_tuple()[1])
"""Python minor version digit (e.g. 9 for 3.9.5) OTVision is currently running with"""

PY_PATCH_VERSION = int(platform.python_version_tuple()[2])
"""Python patch version digit (e.g. 5 for 3.9.5) OTVision is currently running with"""


def _has_cuda() -> bool:
    """Returns True if CUDA is installed on machine

    Returns:
        Bool: If CUDA is installed on machine or not
    """
    import torch

    return torch.cuda.is_available()


def print_has_cuda() -> None:
    """Returns True if CUDA is installed on machine

    Returns:
        Bool: If CUDA is installed on machine or not
    """

    print(f"This system has cuda: {_has_cuda()}")
