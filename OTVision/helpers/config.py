import os
import configparser


# Constants
APPDATA_ROAMING_FOLDER = os.getenv("APPDATA")
USER_SETTINGS_REL_PATH = "OpenTrafficCam\\OTVision\\user_settings.ini"
USER_SETTINGS_PATH = os.path.join(APPDATA_ROAMING_FOLDER, USER_SETTINGS_REL_PATH)


def write_user_settings(config):
    """
    Function to read user settings

    Args:
    config -- configparser element

    Returns: No returns
    """
    with open(USER_SETTINGS_PATH, "w") as configfile:
        config.write(configfile)


def read_user_settings():
    """
    Function to write user settings

    Args: No args

    Returns:
    config -- configparser element
    """
    config = configparser.ConfigParser()
    config.optionxform = lambda option: option
    config.read(USER_SETTINGS_PATH)
    # if os.path.isfile(USER_SETTINGS_PATH):
    if not config.has_section("PATHS"):
        config.add_section("PATHS")
        write_user_settings(config)
    return config


if __name__ == "__main__":
    print("Path of user settings file: " + USER_SETTINGS_PATH)
    config = read_user_settings()
    print("Sections:")
    print(config.sections())
    print("Key-Value pairs:")
    for section in config.sections():
        print(dict(config[section]))
