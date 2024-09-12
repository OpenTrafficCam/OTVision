"""
OTVision gui module for detect.py
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
import tkinter.ttk as ttk

from OTVision.config import CONFIG, PAD
from OTVision.detect.detect import main as detect
from OTVision.helpers.files import get_files, replace_filetype
from OTVision.helpers.log import LOGGER_NAME
from OTVision.view.view_helpers import FrameRun

log = logging.getLogger(LOGGER_NAME)


class FrameDetect(tk.LabelFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_options = FrameDetectOptions(
            master=self
        )  # Always name this "frame_options"
        self.frame_options.pack(**PAD, fill="both", expand=1, anchor="n")
        self.frame_run = FrameRunDetection(master=self)
        self.frame_run.pack(**PAD, side="left", fill="both", expand=1, anchor="s")


class FrameDetectOptions(tk.Frame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Weights
        self.label_weights = tk.Label(master=self, text="Weights")
        self.label_weights.grid(row=0, column=0, sticky="w")
        self.combo_weights = ttk.Combobox(
            master=self, values=CONFIG["DETECT"]["YOLO"]["AVAILABLEWEIGHTS"], width=8
        )
        self.combo_weights.grid(row=0, column=1, sticky="w")
        self.combo_weights.set(CONFIG["DETECT"]["YOLO"]["WEIGHTS"])
        # Confidence
        self.label_conf = tk.Label(master=self, text="Confidence")
        self.label_conf.grid(row=1, column=0, sticky="sw")
        self.scale_conf = tk.Scale(
            master=self, from_=0, to=1, resolution=0.01, orient="horizontal"
        )
        self.scale_conf.grid(row=1, column=1, sticky="w")
        self.scale_conf.set(CONFIG["DETECT"]["YOLO"]["CONF"])
        # IOU
        self.label_iou = tk.Label(master=self, text="IOU")
        self.label_iou.grid(row=2, column=0, sticky="sw")
        self.scale_iou = tk.Scale(
            master=self, from_=0, to=1, resolution=0.01, orient="horizontal"
        )
        self.scale_iou.grid(row=2, column=1, sticky="w")
        self.scale_iou.set(CONFIG["DETECT"]["YOLO"]["IOU"])
        # Image size
        self.label_imgsize = tk.Label(master=self, text="Image size")
        self.label_imgsize.grid(row=3, column=0, sticky="sw")
        self.scale_imgsize = tk.Scale(
            master=self, from_=100, to=1000, resolution=10, orient="horizontal"
        )
        self.scale_imgsize.grid(row=3, column=1, sticky="w")
        self.scale_imgsize.set(CONFIG["DETECT"]["YOLO"]["IMGSIZE"])
        # Chunk size
        self.label_chunksize = tk.Label(master=self, text="Chunk size")
        self.label_chunksize.grid(row=4, column=0, sticky="sw")
        self.scale_chunksize = tk.Scale(
            master=self, from_=1, to=20, resolution=1, orient="horizontal"
        )
        self.scale_chunksize.grid(row=4, column=1, sticky="w")
        self.scale_chunksize.set(CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"])
        # Normalized
        self.checkbutton_normalized_var = tk.BooleanVar()
        self.checkbutton_normalized = tk.Checkbutton(
            master=self, text="Normalized", variable=self.checkbutton_normalized_var
        )
        self.checkbutton_normalized.grid(row=5, column=0, columnspan=2, sticky="w")
        # self.checkbutton_overwrite.select()  # BUG: Still selected
        # Overwrite
        self.checkbutton_overwrite_var = tk.BooleanVar()
        self.checkbutton_overwrite = tk.Checkbutton(
            master=self, text="Overwrite", variable=self.checkbutton_overwrite_var
        )
        self.checkbutton_overwrite.grid(row=6, column=0, columnspan=2, sticky="w")
        if CONFIG["DETECT"]["OVERWRITE"]:
            self.checkbutton_overwrite.select()


class FrameRunDetection(FrameRun):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.button_run.bind("<ButtonRelease-1>", self.run)
        if CONFIG["DETECT"]["RUN_CHAINED"]:
            self.checkbutton_run_chained.select()

    def run(self, event):
        input_filetype = f".{self.master.master.frame_files.combo_vid_filetype.get()}"

        files = replace_filetype(
            files=self.master.master.frame_files.get_tree_files(),
            new_filetype=input_filetype,
        )
        files = get_files(
            paths=files,
            filetypes=[input_filetype],
        )

        weights = self.master.frame_options.combo_weights.get()
        conf = self.master.frame_options.scale_conf.get()
        iou = self.master.frame_options.scale_iou.get()
        size = self.master.frame_options.scale_imgsize.get()
        chunksize = self.master.frame_options.scale_chunksize.get()
        normalized = self.master.frame_options.checkbutton_normalized_var.get()
        overwrite = self.master.frame_options.checkbutton_overwrite_var.get()
        log.info("Call detect from GUI")
        detect(
            paths=files,
            filetypes=[input_filetype],
            weights=weights,
            conf=conf,
            iou=iou,
            size=size,
            chunksize=chunksize,
            normalized=normalized,
            overwrite=overwrite,
        )

        self.master.master.frame_files.update_files_dict()
