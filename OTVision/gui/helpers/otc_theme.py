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


import PySimpleGUI as sg
from config import CONFIG

# Add your new theme colors and settings
sg.LOOK_AND_FEEL_TABLE["OTC"] = {
    "BACKGROUND": "#ffffff",
    "TEXT": "#000000",
    "INPUT": "#ebebeb",
    "TEXT_INPUT": "#000000",
    "SCROLL": "#ebebeb",
    "BUTTON": ("#ffffff", "#37483e"),
    "PROGRESS": ("#37483e", "#adadad"),
    "BORDER": 1,
    "SLIDER_DEPTH": 0,
    "PROGRESS_DEPTH": 0,
}


# Switch to use your newly created theme
OTC_THEME = sg.theme("OTC")


# Official OTC font
sg.SetOptions(font=(CONFIG["GUI"]["FONT"], CONFIG["GUI"]["FONTSIZE"]))
# OTC_FONTTYPE = "bold"


if __name__ == "__main__":
    # Call a popup to show what the theme looks like
    layout = [[sg.T("Test text")], [sg.B("Test button")]]
    window = sg.Window(
        "This is how OTC themed window looks like",
        layout=layout,
        icon=CONFIG["GUI"]["OTC ICON"],
    )
    while True:
        event, values = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED:
            break

    window.close()
