# OTVision: Python module to testwise run OTVision/transform/transform.py

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

from pathlib import Path

from OTVision.transform.transform import main as transform

if __name__ == "__main__":
    transform(
        tracks_files=Path(__file__).parents[0]
        / r"tests"
        / r"data"
        / r"Testvideo_FR20_Cars-Cyclist.ottrk",
        reftpts_file=Path(__file__).parents[0]
        / r"tests"
        / r"data"
        / r"Testvideo_FR20_Cars-Cyclist.otrfpts",
    )
