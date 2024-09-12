"""
OTVision gui module for track.py
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


import logging
import tkinter as tk

from OTVision.config import CONFIG, PAD
from OTVision.helpers.files import get_files, replace_filetype
from OTVision.helpers.log import LOGGER_NAME
from OTVision.track.track import main as track
from OTVision.view.view_helpers import FrameRun

log = logging.getLogger(LOGGER_NAME)


class FrameTrack(tk.LabelFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_options = FrameTrackOptions(
            master=self
        )  # Always name this "frame_options"
        self.frame_options.pack(**PAD, fill="both", expand=1, anchor="n")
        self.frame_run = FrameRunTracking(master=self)
        self.frame_run.pack(**PAD, side="left", fill="both", expand=1, anchor="s")


class FrameTrackOptions(tk.Frame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Sigma l
        self.label_sigma_l = tk.Label(master=self, text="sigma l")
        self.label_sigma_l.grid(row=0, column=0, sticky="sw")
        self.scale_sigma_l = tk.Scale(
            master=self, from_=0, to=1, resolution=0.01, orient="horizontal"
        )
        self.scale_sigma_l.grid(row=0, column=1, sticky="w")
        self.scale_sigma_l.set(CONFIG["TRACK"]["IOU"]["SIGMA_L"])
        # Sigma h
        self.label_sigma_h = tk.Label(master=self, text="sigma h")
        self.label_sigma_h.grid(row=1, column=0, sticky="sw")
        self.scale_sigma_h = tk.Scale(
            master=self, from_=0, to=1, resolution=0.01, orient="horizontal"
        )
        self.scale_sigma_h.grid(row=1, column=1, sticky="w")
        self.scale_sigma_h.set(CONFIG["TRACK"]["IOU"]["SIGMA_H"])
        # Sigma IOU
        self.label_sigma_iou = tk.Label(master=self, text="sigma IOU")
        self.label_sigma_iou.grid(row=2, column=0, sticky="sw")
        self.scale_sigma_iou = tk.Scale(
            master=self, from_=0, to=1, resolution=0.01, orient="horizontal"
        )
        self.scale_sigma_iou.grid(row=2, column=1, sticky="w")
        self.scale_sigma_iou.set(CONFIG["TRACK"]["IOU"]["SIGMA_IOU"])
        # t min
        self.label_t_min = tk.Label(master=self, text="t min")
        self.label_t_min.grid(row=3, column=0, sticky="sw")
        self.scale_t_min = tk.Scale(
            master=self, from_=0, to=20, resolution=1, orient="horizontal"
        )
        self.scale_t_min.grid(row=3, column=1, sticky="w")
        self.scale_t_min.set(CONFIG["TRACK"]["IOU"]["T_MIN"])
        # t miss max
        self.label_t_miss_max = tk.Label(master=self, text="t miss max")
        self.label_t_miss_max.grid(row=4, column=0, sticky="sw")
        self.scale_t_miss_max = tk.Scale(
            master=self, from_=0, to=100, resolution=1, orient="horizontal"
        )
        self.scale_t_miss_max.grid(row=4, column=1, sticky="w")
        self.scale_t_miss_max.set(CONFIG["TRACK"]["IOU"]["T_MISS_MAX"])
        # Overwrite
        self.checkbutton_overwrite_var = tk.BooleanVar()
        self.checkbutton_overwrite = tk.Checkbutton(
            master=self, text="Overwrite", variable=self.checkbutton_overwrite_var
        )
        self.checkbutton_overwrite.grid(row=6, column=0, columnspan=2, sticky="w")
        if CONFIG["TRACK"]["OVERWRITE"]:
            self.checkbutton_overwrite.select()


class FrameRunTracking(FrameRun):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.button_run.bind("<ButtonRelease-1>", self.run)
        if CONFIG["TRACK"]["RUN_CHAINED"]:
            self.checkbutton_run_chained.select()

    def run(self, event):
        files = replace_filetype(
            files=self.master.master.frame_files.get_tree_files(),
            new_filetype=CONFIG["DEFAULT_FILETYPE"]["DETECT"],
        )
        files = get_files(
            paths=files,
            filetypes=[CONFIG["DEFAULT_FILETYPE"]["DETECT"]],
        )
        sigma_l = self.master.frame_options.scale_sigma_l.get()
        sigma_h = self.master.frame_options.scale_sigma_h.get()
        sigma_iou = self.master.frame_options.scale_sigma_iou.get()
        t_min = self.master.frame_options.scale_t_min.get()
        t_miss_max = self.master.frame_options.scale_t_miss_max.get()
        overwrite = self.master.frame_options.checkbutton_overwrite_var.get()
        log.info("Call track from GUI")
        track(
            paths=files,
            sigma_l=sigma_l,
            sigma_h=sigma_h,
            sigma_iou=sigma_iou,
            t_min=t_min,
            t_miss_max=t_miss_max,
            overwrite=overwrite,
        )
        self.master.master.frame_files.update_files_dict()
