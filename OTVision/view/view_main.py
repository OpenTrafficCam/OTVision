import tkinter as tk
import tkinter.ttk as ttk
from view.view_convert import FrameConvert
from view.view_detect import FrameDetect
from view.view_track import FrameTrack
from config import CONFIG

FRAME_WIDTH = 50
pad_options = {"padx": 5, "pady": 5}


class WindowOTVision(tk.Tk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title("OTVision")
        self.iconbitmap(CONFIG["GUI"]["OTC ICON"])
        self.set_layout()

    def set_layout(self):
        # Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, expand=True)
        # Frames
        self.frame_convert = FrameConvert(master=self.notebook)
        self.notebook.add(self.frame_convert, text="Convert")
        self.frame_detect = FrameDetect(master=self.notebook)
        self.notebook.add(self.frame_detect, text="Detect")
        self.frame_track = FrameTrack(master=self.notebook)
        self.notebook.add(self.frame_track, text="Track")
        # self.frame_undistort = FrameConvert(master=self.notebook)
        # self.notebook.add(self.frame_undistort, text="Undistort")
        # self.frame_transform = FrameConvert(master=self.notebook)
        # self.notebook.add(self.frame_transform, text="Transform")


def main():
    app = WindowOTVision()
    app.mainloop()


if __name__ == "__main__":
    main()
