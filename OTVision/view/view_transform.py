"""
OTVision gui module for transform.py
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
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from OTVision.config import CONFIG, PAD
from OTVision.helpers.files import get_files, replace_filetype
from OTVision.helpers.log import LOGGER_NAME
from OTVision.transform.reference_points_picker import ReferencePointsPicker
from OTVision.transform.transform import main as transform
from OTVision.transform.transform import write_refpts
from OTVision.view.view_helpers import FrameRun

log = logging.getLogger(LOGGER_NAME)


class FrameTransform(tk.LabelFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_options = FrameTransformOptions(master=self)
        self.frame_options.pack(**PAD, fill="x", expand=1, anchor="n")
        self.frame_run = FrameRunTransformation(master=self)
        self.frame_run.pack(**PAD, side="left", fill="both", expand=1, anchor="s")


class FrameTransformOptions(tk.Frame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Reference points
        self.button_choose_refpts = tk.Button(
            master=self, text="Choose reference points for selected", state=tk.DISABLED
        )
        self.button_choose_refpts.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.button_choose_refpts.bind("<ButtonRelease-1>", self.choose_refpts)
        self.button_click_refpts = tk.Button(
            master=self, text="New reference points", state=tk.DISABLED
        )
        self.button_click_refpts.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.button_click_refpts.bind("<ButtonRelease-1>", self.click_refpts)

        # Overwrite
        self.checkbutton_overwrite_var = tk.BooleanVar()
        self.checkbutton_overwrite = tk.Checkbutton(
            master=self,
            text="Overwrite existing tracks",
            variable=self.checkbutton_overwrite_var,
        )
        self.checkbutton_overwrite.grid(row=4, column=0, columnspan=2, sticky="w")
        if CONFIG["TRANSFORM"]["OVERWRITE"]:
            self.checkbutton_overwrite.select()

    def choose_refpts(self, event):  # sourcery skip: use-named-expression
        # Get selected files from files frame
        selected_files = self.master.master.frame_files.get_selected_files()

        if selected_files:
            log.debug("choose refpts file for selected files")

            # Show filedialog
            refpts_file = filedialog.askopenfilename(
                title="Select a reference points files",
                filetypes=[(".otrfpts", ".otrfpts")],
                initialdir=Path(selected_files[0]).parent,
            )

            # Check paths
            refpts_file = get_files(paths=[refpts_file])[0]
            log.debug(refpts_file)

            # Copy refpts file for all selected
            for selected_file in selected_files:
                new_refpts_file = Path(selected_file).with_suffix(".otrfpts")
                try:
                    shutil.copy2(refpts_file, new_refpts_file)
                except shutil.SameFileError:
                    continue

            # Update dict and treeview in files frame
            self.master.master.frame_files.update_files_dict()

    def click_refpts(self, event):
        if selected_files := self.master.master.frame_files.get_selected_files():
            log.debug("click and save refpts for selected files")

            if refpts := ReferencePointsPicker(
                video_file=Path(selected_files[0])
            ).refpts:
                # Save refpts for all selected files
                for selected_file in selected_files:
                    new_refpts_file = Path(selected_file).with_suffix(".otrfpts")
                    write_refpts(refpts=refpts, refpts_file=new_refpts_file)

                # Update dict and treeview in files frame
                self.master.master.frame_files.update_files_dict()
        else:
            log.debug("No files selected")


class FrameRunTransformation(FrameRun):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.button_run.bind("<ButtonRelease-1>", self.run)
        if CONFIG["TRANSFORM"]["RUN_CHAINED"]:
            self.checkbutton_run_chained.select()

    def run(self, event):
        tracks_files = replace_filetype(
            files=self.master.master.frame_files.get_tree_files(),
            new_filetype=CONFIG["DEFAULT_FILETYPE"]["TRACK"],
        )
        tracks_files = get_files(
            paths=tracks_files,
            filetypes=[CONFIG["DEFAULT_FILETYPE"]["TRACK"]],
        )
        log.info("Call transform from GUI")
        transform(paths=tracks_files)
