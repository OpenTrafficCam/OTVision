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

import os
import configparser


def get_user_settings_path_appdata():
    APPDATA_ROAMING_FOLDER = os.getenv("APPDATA")
    USER_SETTINGS_REL_PATH = "OpenTrafficCam\\OTVision\\user_settings.ini"
    USER_SETTINGS_PATH = os.path.join(APPDATA_ROAMING_FOLDER, USER_SETTINGS_REL_PATH)
    return USER_SETTINGS_PATH


def get_user_settings_path():
    USER_SETTINGS_REL_PATH = "OTVision\\helpers\\user.conf"
    return USER_SETTINGS_REL_PATH


def write_user_settings(config):
    """
    Function to read user settings

    Args:
    config -- configparser element

    Returns: No returns
    """
    USER_SETTINGS_PATH = get_user_settings_path()
    with open(USER_SETTINGS_PATH, "w") as configfile:
        config.write(configfile)


def read_user_settings():
    """
    Function to write user settings

    Args: No args

    Returns:
    config -- configparser element
    """
    USER_SETTINGS_PATH = get_user_settings_path()
    config = configparser.ConfigParser()
    config.optionxform = lambda option: option
    config.read(USER_SETTINGS_PATH)
    # if os.path.isfile(USER_SETTINGS_PATH):
    if not config.has_section("PATHS"):
        config.add_section("PATHS")
        write_user_settings(config)
    return config


if __name__ == "__main__":
    config = read_user_settings()
    print("Sections:")
    print(config.sections())
    print("Key-Value pairs:")
    for section in config.sections():
        print(dict(config[section]))
