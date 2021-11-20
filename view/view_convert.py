import tkinter as tk
import tkinter.ttk as ttk
from view.view_helpers import FrameFiles, FrameSubmit, FrameGoTo
from config import CONFIG

# BUG: Space at bottom of all LabelFrames


class FrameConvert(tk.Frame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_videos = FrameFiles(
            master=self,
            text="Choose h264 files",
            filecategory="h264 videos",
            default_filetype=".h264",
        )
        self.frame_videos.pack(fill="both", expand=1)
        self.frame_options = FrameConvertOptions(master=self, text="Configure")
        self.frame_options.pack(fill="both", expand=1)
        self.frame_submit = FrameSubmit(master=self, text="Start conversion")
        self.frame_submit.pack(fill="both", expand=1)
        self.frame_goto = FrameGoTo(
            master=self, text="Continue with next step", text_button="Go to detection!"
        )
        self.frame_goto.pack(fill="both", expand=1)


class FrameConvertOptions(tk.LabelFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # File type
        self.label_filtype = tk.Label(master=self, text="Output file type")
        self.label_filtype.grid(row=0, column=0, sticky="w")
        self.combo_filtype = ttk.Combobox(
            master=self,
            values=["avi", "mov", "m4v", "mkv", "mp4", "mpeg", "mpg", "wmv"],
        )  # TODO: Get from config
        self.combo_filtype.grid(row=0, column=1, sticky="w")
        self.combo_filtype.current(4)
        # Frame rate from video
        self.checkbutton_use_framerate = tk.Checkbutton(
            master=self, text="Use frame rate from file name"
        )
        self.checkbutton_use_framerate.grid(row=1, column=0, columnspan=2, sticky="w")
        self.checkbutton_use_framerate.select()
        # Input frame rate  # TODO: Show only when previous checkbutton not selected
        self.label_framerate = tk.Label(master=self, text="Input frame rate")
        self.label_framerate.grid(row=2, column=0, sticky="w")
        self.entry_framerate = tk.Entry(master=self)
        self.entry_framerate.grid(row=2, column=1, sticky="w")
        self.entry_framerate.insert(index=-1, string="25.0")  # TODO: Get from config
        # Overwrite
        self.checkbutton_overwrite = tk.Checkbutton(
            master=self, text="Overwrite existing videos"
        )
        self.checkbutton_overwrite.grid(row=3, column=0, columnspan=2, sticky="w")
        self.checkbutton_overwrite.select()
