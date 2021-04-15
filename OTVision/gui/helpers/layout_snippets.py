# OTVision: Python gui snippets

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
from config import CONFIG


def _frame_folders_files(
    title="Choose folders or files", files=[], width=CONFIG["GUI"]["FRAMEWIDTH"]
):
    button_browse_folders_files = sg.Button(
        "Browse files and/or folders", key="-button_browse_folders_files-"
    )
    text_files = sg.Text(
        "Number of selected files: " + str(len(files)),
        key="-text_files-",
    )
    frame_folders_files = sg.Frame(
        title=title,
        layout=[
            [sg.Text("", size=(width, 1))],
            [button_browse_folders_files],
            [text_files],
            [sg.Text("", size=(width, 1))],
        ],
        element_justification="center",
    )
    return frame_folders_files
