import tkinter as tk
import tkinter.ttk as ttk
from view.view_helpers import FrameFiles, FrameSubmit, FrameGoTo, FrameSubmit
from config import CONFIG
from convert.convert import main as convert

# BUG: Space at bottom of all LabelFrames


class FrameConvert(tk.Frame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.frame_files = FrameFiles(
        #     master=self,
        #     text="Choose h264 files",
        #     filecategory="h264 videos",
        #     default_filetype=".h264",
        # )
        # self.frame_files.pack(fill="both", expand=1)
        self.frame_options = FrameConvertOptions(master=self, text="Configure")
        self.frame_options.pack(fill="both", expand=1)
        self.frame_submit = FrameStartConversion(
            master=self, text="Start conversion", button_label="Convert"
        )
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
        self.checkbutton_use_framerate_var = tk.BooleanVar()
        self.checkbutton_use_framerate = tk.Checkbutton(
            master=self,
            text="Use frame rate from file name",
            variable=self.checkbutton_use_framerate_var,
        )
        self.checkbutton_use_framerate.bind(
            "<ButtonRelease-1>", self.toggle_entry_framerate
        )
        self.checkbutton_use_framerate.grid(row=1, column=0, columnspan=2, sticky="w")
        self.checkbutton_use_framerate.select()
        # Input frame rate  # TODO: Show only when previous checkbutton not selected
        self.label_framerate = tk.Label(master=self, text="Input frame rate")
        self.label_framerate.grid(row=2, column=0, sticky="w")
        self.entry_framerate = tk.Entry(master=self)
        self.entry_framerate.grid(row=2, column=1, sticky="w")
        self.entry_framerate.insert(index=0, string="20.0")  # TODO: Get from config
        self.entry_framerate.configure(state="disabled")
        # Overwrite
        self.checkbutton_overwrite = tk.Checkbutton(
            master=self, text="Overwrite existing videos"
        )
        self.checkbutton_overwrite.grid(row=3, column=0, columnspan=2, sticky="w")
        self.checkbutton_overwrite.select()

    def toggle_entry_framerate(self, event):
        if self.checkbutton_use_framerate_var.get():
            self.entry_framerate.configure(state="normal")
            self.entry_framerate.delete(0, "end")
            self.entry_framerate.insert(0, 20.0)
        else:
            self.entry_framerate.configure(state="disabled")


class FrameStartConversion(FrameSubmit):
    def __init__(self, button_label="Submit", **kwargs):
        super().__init__(**kwargs)
        self.button_submit.bind("<ButtonRelease-1>", self.submit)

    def submit(self, event):
        print(self.master.frame_options.checkbutton_use_framerate_var.get())
        paths = list(self.master.frame_files.get_listbox_files())
        output_filetype = "." + self.master.frame_options.combo_filtype.get()
        input_fps = self.master.frame_options.entry_framerate.get()
        output_fps = self.master.frame_options.entry_framerate.get()
        overwrite = self.master.frame_options.checkbutton_use_framerate_var.get()
        # print(paths)
        # print(output_filetype)
        # print(input_fps)
        # print(output_fps)
        # print(overwrite)
        convert(
            paths=paths,
            output_filetype=output_filetype,
            input_fps=input_fps,
            output_fps=output_fps,
            fps_from_filename=True,
            overwrite=overwrite,
        )
        print("Conversion succesful")


class FrameGoToDetect(FrameGoTo):
    def __init__(self, text_button="Submit", **kwargs):
        super().__init__(**kwargs)
