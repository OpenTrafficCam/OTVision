import tkinter as tk
import tkinter.ttk as ttk

from numpy import size
from view.view_helpers import FrameFiles, FrameRun, FrameGoTo
from config import CONFIG, PAD
from detect.detect import main as detect
from helpers.files import get_files


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
        self.scale_conf.set(0.25)  # TODO: Get from config
        # IOU
        self.label_iou = tk.Label(master=self, text="IOU")
        self.label_iou.grid(row=2, column=0, sticky="sw")
        self.scale_iou = tk.Scale(
            master=self, from_=0, to=1, resolution=0.01, orient="horizontal"
        )
        self.scale_iou.grid(row=2, column=1, sticky="w")
        self.scale_iou.set(0.45)  # TODO: Get from config
        # Image size
        self.label_imgsize = tk.Label(master=self, text="Image size")
        self.label_imgsize.grid(row=3, column=0, sticky="sw")
        self.scale_imgsize = tk.Scale(
            master=self, from_=100, to=1000, resolution=10, orient="horizontal"
        )
        self.scale_imgsize.grid(row=3, column=1, sticky="w")
        self.scale_imgsize.set(640)  # TODO: Get from config
        # Chunk size
        self.label_chunksize = tk.Label(master=self, text="Chunk size")
        self.label_chunksize.grid(row=4, column=0, sticky="sw")
        self.scale_chunksize = tk.Scale(
            master=self, from_=1, to=20, resolution=1, orient="horizontal"
        )
        self.scale_chunksize.grid(row=4, column=1, sticky="w")
        self.scale_chunksize.set(1)  # TODO: Get from config
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
        self.checkbutton_overwrite.select()


class FrameRunDetection(FrameRun):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.button_run.bind("<ButtonRelease-1>", self.run)

    def run(self, event):
        print("---Starting detection---")
        input_filetype = "." + self.master.master.frame_files.combo_vid_filetype.get()
        paths = get_files(
            paths=self.master.master.frame_files.get_tree_files(),
            filetypes=input_filetype,
            replace_filetype=True,
        )
        weights = self.master.frame_options.combo_weights.get()
        conf = self.master.frame_options.scale_conf.get()
        iou = self.master.frame_options.scale_iou.get()
        size = self.master.frame_options.scale_imgsize.get()
        chunksize = self.master.frame_options.scale_chunksize.get()
        normalized = self.master.frame_options.checkbutton_normalized_var.get()
        overwrite = self.master.frame_options.checkbutton_overwrite_var.get()
        detect(
            files=paths,
            filetype=input_filetype,
            weights=weights,
            conf=conf,
            iou=iou,
            size=size,
            chunksize=chunksize,
            normalized=normalized,
        )  # TODO: Add overwrite parameter (first add argument to detect)
        self.master.master.frame_files.update_files_dict()
        print("---Detection successful---")
