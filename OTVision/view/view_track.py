import tkinter as tk
import tkinter.ttk as ttk
from view.view_helpers import FrameFiles, FrameRun, FrameGoTo
from config import CONFIG, PAD
from track.track import main as track
from helpers.files import get_files


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
        self.label_sigma_l.grid(row=0, column=0, sticky="w")
        self.scale_sigma_l = tk.Scale(
            master=self, from_=0, to=1, resolution=0.01, orient="horizontal"
        )
        self.scale_sigma_l.grid(row=0, column=1, sticky="w")
        self.scale_sigma_l.set(0.25)  # TODO: Get from config
        # Sigma h
        self.label_sigma_h = tk.Label(master=self, text="sigma h")
        self.label_sigma_h.grid(row=1, column=0, sticky="w")
        self.scale_sigma_h = tk.Scale(
            master=self, from_=0, to=1, resolution=0.01, orient="horizontal"
        )
        self.scale_sigma_h.grid(row=1, column=1, sticky="w")
        self.scale_sigma_h.set(0.8)  # TODO: Get from config
        # Sigma IOU
        self.label_sigma_iou = tk.Label(master=self, text="sigma IOU")
        self.label_sigma_iou.grid(row=2, column=0, sticky="w")
        self.scale_sigma_iou = tk.Scale(
            master=self, from_=0, to=1, resolution=0.01, orient="horizontal"
        )
        self.scale_sigma_iou.grid(row=2, column=1, sticky="w")
        self.scale_sigma_iou.set(0.3)  # TODO: Get from config
        # t min
        self.label_t_min = tk.Label(master=self, text="t min")
        self.label_t_min.grid(row=3, column=0, sticky="w")
        self.scale_t_min = tk.Scale(
            master=self, from_=0, to=20, resolution=1, orient="horizontal"
        )
        self.scale_t_min.grid(row=3, column=1, sticky="w")
        self.scale_t_min.set(5)  # TODO: Get from config
        # t miss max
        self.label_t_miss_max = tk.Label(master=self, text="t miss max")
        self.label_t_miss_max.grid(row=4, column=0, sticky="w")
        self.scale_t_miss_max = tk.Scale(
            master=self, from_=0, to=10, resolution=1, orient="horizontal"
        )
        self.scale_t_miss_max.grid(row=4, column=1, sticky="w")
        self.scale_t_miss_max.set(10)  # TODO: Get from config
        # Overwrite
        self.checkbutton_overwrite_var = tk.BooleanVar()
        self.checkbutton_overwrite = tk.Checkbutton(
            master=self, text="Overwrite", variable=self.checkbutton_overwrite_var
        )
        self.checkbutton_overwrite.grid(row=6, column=0, columnspan=2, sticky="w")
        self.checkbutton_overwrite.select()


class FrameRunTracking(FrameRun):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.button_run.bind("<ButtonRelease-1>", self.run)

    def run(self, event):
        print("---Starting tracking---")
        paths = get_files(
            paths=self.master.master.frame_files.get_tree_files(),
            filetypes=CONFIG["DEFAULT_FILETYPE"]["DETECT"],
            replace_filetype=True,
        )
        sigma_l = self.master.frame_options.scale_sigma_l.get()
        sigma_h = self.master.frame_options.scale_sigma_h.get()
        sigma_iou = self.master.frame_options.scale_sigma_iou.get()
        t_min = self.master.frame_options.scale_t_min.get()
        t_miss_max = self.master.frame_options.scale_t_miss_max.get()
        overwrite = self.master.frame_options.checkbutton_overwrite_var.get()
        track(
            paths=paths,
            sigma_l=sigma_l,
            sigma_h=sigma_h,
            sigma_iou=sigma_iou,
            t_min=t_min,
            t_miss_max=t_miss_max,
            overwrite=overwrite,
        )
        self.master.master.frame_files.update_files_dict()
        print("---Tracking successful---")
