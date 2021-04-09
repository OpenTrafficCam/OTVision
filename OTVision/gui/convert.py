# OTVision: Python gui to convert h264 based videos to other formats and frame rates

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


import PySimpleGUI as sg
from PySimpleGUI.PySimpleGUI import DEFAULT_TEXT_JUSTIFICATION, Text
from pathlib import Path

from gui.helpers import browse_folders_and_files
from gui.helpers.sg_otc_theme import (
    OTC_ICON,
    OTC_THEME,
)
from config import CONFIG
from convert.convert import main as convert
from helpers.files import get_files


def main(sg_theme=OTC_THEME):
    folders = CONFIG["LAST PATHS"]["FOLDER"]
    single_files = CONFIG["LAST PATHS"]["VIDEO"]
    files = get_files(
        paths=[*folders, *single_files],
        filetypes=CONFIG["FILETYPES"]["VID"].append(".h264"),
    )
    sg.SetOptions(font=(CONFIG["GUI"]["FONT"], CONFIG["GUI"]["FONTSIZE"]))

    # Get initial layout and create initial window
    layout, text_status_detect = create_layout(files)
    window = sg.Window(
        title="OTVision: Detect",
        layout=layout,
        icon=OTC_ICON,
        location=(
            CONFIG["GUI"]["WINDOW"]["LOCATION_X"],
            CONFIG["GUI"]["WINDOW"]["LOCATION_Y"],
        ),
        resizable=True,
        finalize=True,
    )
    window.maximize()

    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read(timeout=0)
        # Close Gui
        if (
            event == sg.WIN_CLOSED or event == "Cancel" or event == "-BUTTONBACKTOHOME-"
        ):  # if user closes window or clicks cancel
            break

    window.close()


def create_layout(files):
    # Gui elements
    button_browse_folders_files = sg.Button(
        "Browse files and/or folders", key="-button_browse_folders_files-"
    )
    input_fps = sg.In(CONFIG["CONVERT"]["FPS"], enable_events=True)
    drop_output_filetype = sg.Drop(
        [CONFIG["FILETYPES"]["VID"].append(".h264")],
        default_value=CONFIG["CONVERT"]["OUTPUT_FILETYPE"],
    )
    check_overwrite = sg.Check("Overwrite", default=CONFIG["CONVERT"]["OVERWRITE"])

    # Create layout
    layout = [
        [button_browse_folders_files],
        [drop_output_filetype, input_fps, check_overwrite],
    ]

    return layout


if __name__ == "__main__":
    pass
