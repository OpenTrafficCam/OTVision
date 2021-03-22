# OTVision: Python module to read and write configuration dict

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
import configparser
import json


def read(config_name: str = "user"):
    # TODO: Also read default.otconf and compare to fill up missing stuff in user config
    config_path = get_path(config_name=config_name)
    if config_path.suffix == ".otconf":
        if not config_path.is_file():
            print(f"{config_name}.otconf doesnt exist, load default.otconf instead")
            config_path = get_path(config_name="default")
            if not config_path.is_file():
                raise FileNotFoundError()
        with open(str(config_path)) as f:
            config = json.load(f)
        return config
    else:
        raise ValueError("Filetype for configuratuin has to be .otconf")


def write(config: dict, config_name: str = "user"):
    config_path = get_path(config_name=config_name)
    if config_name == "default":
        answer = input("Sure you wanna overwrite default.otconf? [y/n]")
        if answer != "y":
            print("Configuration not saved, default.otconf not overwritten")
            return None
        print("default.otconf overwritten")
    with open(str(config_path), "w") as f:
        json.dump(config, f, indent=4)


def get_path(config_name="default"):
    config_path = Path(__file__).parents[0] / f"{config_name}.otconf"
    return config_path


def write_configparser(config_dict: dict, config_type: str = "defaulttest"):
    config_path = get_path(config_type=config_type)
    if config_path.is_file():
        overwrite = True
    config = configparser.ConfigParser()
    try:
        for section, subdict in config_dict.items:
            config[section] = {}
            for key, value in subdict.items:
                config[section][key] = str(value)
    except TypeError("write_dict needs nested dict of configuration"):
        return None
    # Write configuration dict to ".conf" file using configparser
    with open(str(config_path), "w") as configfile:
        config.write(configfile)
    if overwrite:
        print(f"{config_type} conf file overwritten")
    else:
        print(f"{config_type} conf file created")


def read_configparser(config_type: str = "default"):
    """read configuration dict from ".conf" file using configparser

    Args:
        config_type (str, optional): [description]. Defaults to "default".

    Raises:
        FileNotFoundError: If config file of given type does not exist

    Returns:
        dict:
    """
    config_path = get_path(config_type=config_type)
    if config_path.is_file():
        config = configparser.ConfigParser()
        config.optionxform = lambda option: option
        config.read(str(config_path))
        return config._sections
    elif config_type != "default":
        config = configparser.ConfigParser()
        config.optionxform = lambda option: option
        config.read(str(get_path(config_type="default")))
        print(f"{config_type} conf file doesnt exist, default conf file loaded instead")
        return config._sections
    else:
        raise FileNotFoundError("No config file exists")


if __name__ == "__main__":
    config_dict = read()
    print(f"Config dict: {config_dict}")
    write(config_dict, config_name="user")
