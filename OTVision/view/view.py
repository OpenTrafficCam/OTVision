"""
OTVision main gui module
"""

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


import tkinter as tk

from OTVision.config import CONFIG, PAD
from OTVision.helpers.machine import ON_LINUX, ON_WINDOWS
from OTVision.view.view_convert import FrameConvert
from OTVision.view.view_detect import FrameDetect
from OTVision.view.view_helpers import FrameFileTree, FrameRunChained
from OTVision.view.view_track import FrameTrack
from OTVision.view.view_transform import FrameTransform

FRAME_WIDTH = 50


class WindowOTVision(tk.Tk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title("OTVision")
        if ON_WINDOWS:
            self.iconbitmap(CONFIG["GUI"]["OTC ICON"])
        self.set_layout()
        self.minsize(900, 620)
        if ON_LINUX:
            self.state("normal")
        else:
            self.state("zoomed")

    def set_layout(self):
        for col in range(3):
            self.columnconfigure(index=col, weight=1)
        self.rowconfigure(index=0, weight=1)  # Stretches frame_files only
        # Treeview files
        self.frame_files = FrameFileTree(master=self, text="Choose files")
        self.frame_files.grid(**PAD, row=0, column=0, columnspan=4, sticky="nsew")
        # Settings
        # Convert
        self.frame_convert = FrameConvert(master=self, text="Convert")
        self.frame_convert.grid(**PAD, row=1, column=0, sticky="nsew")
        # Detect
        self.frame_detect = FrameDetect(master=self, text="Detect")
        self.frame_detect.grid(**PAD, row=1, column=1, sticky="nsew")
        # Track
        self.frame_track = FrameTrack(master=self, text="Track")
        self.frame_track.grid(**PAD, row=1, column=2, sticky="nsew")
        # # Undistort # TODO
        # self.frame_undistort = FrameUndistort(master=self, text="Undistort")
        # self.frame_undistort.pack(**PAD, side="left", expand=True)
        # # Transform # TODO
        self.frame_transform = FrameTransform(master=self, text="Transform")
        self.frame_transform.grid(**PAD, row=1, column=3, sticky="nsew")
        # Run chained
        self.frame_run_chained = FrameRunChained(master=self, text="Run chained")
        self.frame_run_chained.grid(**PAD, row=2, column=0, columnspan=4, sticky="ew")

    def toggle_frame_detect(self, event):
        if self.checkbutton_convert_var.get():
            self.frame_convert.configure(state="normal")
        else:
            self.frame_convert.configure(state="disabled")


def main():
    global app
    app = WindowOTVision()
    app.mainloop()


if __name__ == "__main__":
    main()
