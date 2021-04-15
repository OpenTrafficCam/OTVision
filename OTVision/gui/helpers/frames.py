# OTVision: Gui classes: Frames

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


class OTFrameFoldersFiles(sg.Frame):
    def __init__(
        self,
        title="Step 1: Browse folders and files",
        files=[],
        width=CONFIG["GUI"]["FRAMEWIDTH"],
    ):
        super().__init__(title=title, layout=self.get_layout(files, width))
        self.ElementJustification = "center"

    def get_layout(self, files, width):  # sourcery skip
        button_browse_folders_files = sg.Button(
            "Browse files and/or folders", key="-button_browse_folders_files-"
        )
        text_files = sg.Text(
            "Number of selected files: " + str(len(files)),
            key="-text_files-",
        )
        layout = [
            [sg.Text("", size=(width, 1))],
            [button_browse_folders_files],
            [text_files],
            [sg.Text("", size=(width, 1))],
        ]
        return layout
