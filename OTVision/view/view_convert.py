import tkinter as tk
import tkinter.ttk as ttk
from view.view_helpers import FrameFiles, FrameRun, FrameGoTo
from config import CONFIG, PAD
from convert.convert import main as convert
from helpers.files import get_files


class FrameConvert(tk.LabelFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_options = FrameConvertOptions(master=self)
        self.frame_options.pack(**PAD, fill="x", expand=1, anchor="n")
        self.frame_run = FrameRunConversion(master=self)
        self.frame_run.pack(**PAD, side="left", fill="both", expand=1, anchor="s")


class FrameConvertOptions(tk.Frame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # File type
        self.label_filtype = tk.Label(master=self, text="Output file type")
        self.label_filtype.grid(row=0, column=0, sticky="w")
        self.combo_filtype = ttk.Combobox(
            master=self,
            values=[str.replace(".", "") for str in CONFIG["FILETYPES"]["VID"]],
            width=5,
        )
        self.combo_filtype.grid(row=0, column=1, sticky="w")
        self.combo_filtype.set(CONFIG["DEFAULT_FILETYPE"]["VID"].replace(".", ""))
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
        # Input frame rate
        self.label_framerate = tk.Label(master=self, text="Input frame rate")
        self.label_framerate.grid(row=2, column=0, sticky="w")
        self.entry_framerate = tk.Entry(master=self, width=4)
        self.entry_framerate.grid(row=2, column=1, sticky="w")
        self.entry_framerate.insert(index=0, string=CONFIG["CONVERT"]["FPS"])
        self.entry_framerate.configure(state="disabled")
        # Overwrite
        self.checkbutton_overwrite_var = tk.BooleanVar()
        self.checkbutton_overwrite = tk.Checkbutton(
            master=self,
            text="Overwrite existing videos",
            variable=self.checkbutton_overwrite_var,
        )
        self.checkbutton_overwrite.grid(row=3, column=0, columnspan=2, sticky="w")
        if CONFIG["CONVERT"]["OVERWRITE"]:
            self.checkbutton_overwrite.select()

    def toggle_entry_framerate(self, event):
        if self.checkbutton_use_framerate_var.get():
            self.entry_framerate.configure(state="normal")
            self.entry_framerate.delete(0, "end")
            self.entry_framerate.insert(0, CONFIG["CONVERT"]["FPS"])
        else:
            self.entry_framerate.configure(state="disabled")


class FrameRunConversion(FrameRun):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.button_run.bind("<ButtonRelease-1>", self.run)

    def run(self, event):
        print("---Starting conversion---")
        fps_from_filename = (
            self.master.frame_options.checkbutton_use_framerate_var.get()
        )
        paths = get_files(
            paths=self.master.master.frame_files.get_tree_files(),
            filetypes=".h264",
            replace_filetype=True,
        )
        output_filetype = "." + self.master.frame_options.combo_filtype.get()
        input_fps = self.master.frame_options.entry_framerate.get()
        output_fps = self.master.frame_options.entry_framerate.get()
        overwrite = self.master.frame_options.checkbutton_use_framerate_var.get()
        convert(
            paths=paths,
            output_filetype=output_filetype,
            input_fps=input_fps,
            output_fps=output_fps,
            fps_from_filename=fps_from_filename,
            overwrite=overwrite,
        )
        self.master.master.frame_files.update_files_dict()
        print("---Conversion successful---")
