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

from gui.helpers.browse_folders_and_files import main as browse_folders_and_files
from gui.helpers.frames import OTFrameFoldersFiles
from gui.helpers.texts import OTTextSpacer
from gui.helpers.sg_otc_theme import (
    OTC_ICON,
    OTC_THEME,
)
from gui.helpers import layout_snippets
from config import CONFIG
from convert.convert import main as convert
from helpers.files import get_files


def main(sg_theme=OTC_THEME):
    folders = CONFIG["LAST PATHS"]["FOLDERS"]
    single_files = CONFIG["LAST PATHS"]["VIDEOS"]
    files = get_files(
        paths=[*folders, *single_files],
        filetypes=[*CONFIG["FILETYPES"]["VID"], ".h264"],
    )
    sg.SetOptions(font=(CONFIG["GUI"]["FONT"], CONFIG["GUI"]["FONTSIZE"]))

    # Get initial layout and create initial window
    layout = create_layout(files)
    window = sg.Window(
        title="OTVision: Convert",
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
        if event in [sg.WIN_CLOSED, "Cancel", "-BUTTONBACKTOHOME-"]:
            break
        # Folders and files
        if event == "-button_browse_folders_files-":
            # TODO: Maybe decide for only file list
            folders, single_files = browse_folders_and_files(
                "Select video files for format conversion",
                filetypes=[*CONFIG["FILETYPES"]["VID"], ".h264"],
                input_files=single_files,
                input_folders=folders,
            )

    window.close()


def create_layout(files):

    # GUI elements: Choose videos
    frame_folders_files = OTFrameFoldersFiles()

    # GUI elements: Set parameters
    width_c1 = 40
    width_c2 = 5
    text_output_filetype = sg.T(
        "Output filetype", justification="right", size=(width_c1, 1)
    )
    drop_output_filetype = sg.Drop(
        [*CONFIG["FILETYPES"]["VID"]],
        default_value=CONFIG["CONVERT"]["OUTPUT_FILETYPE"],
        key="-drop_output_filetype-",
    )
    text_fps_from_input_video = sg.T(
        "Try to use framerate from input video",
        justification="right",
        size=(width_c1, 1),
    )
    check_fps_from_input_video = sg.Check("", default=True)
    text_input_fps = sg.T("Input framerate", justification="right", size=(width_c1, 1))
    in_input_fps = sg.In(
        CONFIG["CONVERT"]["FPS"],
        key="-in_input_fps-",
        enable_events=True,
        size=(width_c2, 10),
    )
    text_output_fps = sg.T(
        "Output framerate", justification="right", size=(width_c1, 1)
    )
    in_output_fps = sg.In(
        CONFIG["CONVERT"]["FPS"],
        key="-in_output_fps-",
        enable_events=True,
        size=(width_c2, 10),
    )
    text_overwrite = sg.T(
        "Overwrite existing videos",
        justification="right",
        size=(width_c1, 1),
    )
    check_overwrite = sg.Check("", default=CONFIG["CONVERT"]["OVERWRITE"])
    frame_parameters = sg.Frame(
        "Step 2: Set parameters",
        [
            [OTTextSpacer()],
            [text_output_filetype, drop_output_filetype],
            [text_fps_from_input_video, check_fps_from_input_video],
            [text_input_fps, in_input_fps],
            [text_output_fps, in_output_fps],
            [text_overwrite, check_overwrite],
            [OTTextSpacer()],
        ],
    )

    # Gui elements: Convert
    button_convert = sg.B("Convert!", key="-button_convert-")
    frame_convert = sg.Frame(
        "Step 3: Start conversion",
        [
            [OTTextSpacer()],
            [button_convert],
            [OTTextSpacer()],
        ],
        size=(400, 10),
        element_justification="center",
    )

    # Create layout
    layout = [[frame_folders_files], [frame_parameters], [frame_convert]]

    return layout


if __name__ == "__main__":
    pass
