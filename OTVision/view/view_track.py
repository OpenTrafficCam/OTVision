import tkinter as tk
import tkinter.ttk as ttk
from view.view_helpers import FrameFiles, FrameSubmit, FrameGoTo
from config import CONFIG


class FrameTrack(tk.Frame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_files = FrameFiles(
            master=self,
            text="Choose detection files",
            filecategory="detection files",
            default_filetype=CONFIG["DEFAULT_FILETYPE"]["DETECT"],
            filetypes=CONFIG["FILETYPES"]["DETECT"],
        )
        self.frame_files.pack(fill="both", expand=1)
        self.frame_options = FrameTrackOptions(
            master=self, text="Configure"
        )  # Always name this "frame_options"
        self.frame_options.pack(fill="both", expand=1)
        self.frame_submit = FrameStartTracking(
            master=self, text="Start tracking", button_label="Track"
        )
        self.frame_submit.pack(fill="both", expand=1)
        # self.frame_goto.pack(fill="both", expand=1)


class FrameTrackOptions(tk.LabelFrame):
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
        self.checkbutton_overwrite = tk.Checkbutton(master=self, text="Overwrite")
        self.checkbutton_overwrite.grid(row=6, column=0, columnspan=2, sticky="w")
        self.checkbutton_overwrite.select()


class FrameStartTracking(FrameSubmit):
    def __init__(self, button_label="Submit", **kwargs):
        super().__init__(**kwargs)
        self.button_submit.bind("<ButtonRelease-1>", self.submit)

    def submit(self, event):
        print("Starting tracking")  #TODO: Call track with parameters