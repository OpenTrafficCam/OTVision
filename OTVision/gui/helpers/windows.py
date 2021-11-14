# OTVision: Gui classes: Windows

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

from OTVision.config import CONFIG


class OTSubpackageWindow(sg.Window):
    def __init__(self, title, layout):
        super().__init__(
            title=title,
            layout=layout,
            icon=CONFIG["GUI"]["OTC ICON"],
            location=(
                CONFIG["GUI"]["WINDOW"]["LOCATION_X"],
                CONFIG["GUI"]["WINDOW"]["LOCATION_Y"],
            ),
            element_justification="center",
            resizable=True,
            finalize=True,
        )
        self.maximize()
