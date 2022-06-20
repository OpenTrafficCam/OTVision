
import tkinter as tk

from OTVision.config import CONFIG, PAD
from OTVision.helpers.machine import ON_WINDOWS
from OTVision.view.view_convert import FrameConvert, FrameConvertDummy
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

    def set_layout(self):
        for col in range(3):
            self.columnconfigure(index=col, weight=1)
        # Treeview files
        self.frame_files = FrameFileTree(master=self, text="Choose files")
        self.frame_files.grid(**PAD, row=0, column=0, columnspan=4, sticky="ew")
        # Settings
        # Convert (Only works on windows machines for now)
        if ON_WINDOWS:
            self.frame_convert = FrameConvert(master=self, text="Convert")
            self.frame_convert.grid(**PAD, row=1, column=0, sticky="nsew")
        else:
            self.frame_convert_dummy = FrameConvertDummy(master=self, text="Convert")
            self.frame_convert_dummy.grid(**PAD, row=1, column=0, sticky="nsew")
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
