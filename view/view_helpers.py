import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog


class FrameFiles(tk.LabelFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # File list
        self.file_list = []  # ?: Add test data?
        self.filecategory = "video files"
        self.filetypes = [".mp4", ".avi"]
        # File buttons
        self.button_add_folder = tk.Button(master=self, text="Add folder")
        self.button_add_folder.bind(
            "<Button-1>", self.select_folders
        )  # BUG: takes 1 positional argument but 2 were given
        self.button_add_folder.grid(row=0, column=0, sticky="ew")
        self.button_add_file = tk.Button(master=self, text="Add file")
        self.button_add_file.bind(
            "<Button-1>", self.select_files
        )  # BUG: takes 1 positional argument but 2 were given
        self.button_add_file.grid(row=0, column=1, sticky="ew")
        self.button_rem_sel = tk.Button(master=self, text="Remove selected")
        self.button_rem_sel.grid(row=0, column=2, sticky="ew")
        self.button_rem_all = tk.Button(master=self, text="Remove all")
        self.button_rem_all.grid(row=0, column=3, sticky="ew")
        # File names
        # self.label_video = tk.Label(master=self, text="Videos to convert:")
        # self.label_video.grid(row=1, column=0, columnspan=4, sticky="w")
        self.listbox_video = tk.Listbox(master=self, width=50)
        self.listbox_video.grid(row=2, column=0, columnspan=4, sticky="ew")

    def select_files(self):
        self.file_list.extend(
            list(
                filedialog.askopenfilenames(
                    title=f"Select one or multiple {self.filecategory}",
                    filetypes=[
                        (self.filecategory, self.filetypes),
                        ("all files", "*.*"),
                    ],
                )
            )
        )

    def select_folders(self):
        self.file_list.extend(
            list(filedialog.askdirectory(title="Select one or multiple folders"))
        )


class FrameSubmit(tk.LabelFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Convert
        self.button_convert = tk.Button(master=self, text="Convert!")
        self.button_convert.grid(row=0, column=0, sticky="ew")
        # Progress bar
        self.progress = ttk.Progressbar(master=self)
        self.progress.grid(row=1, column=0, sticky="ew")


class FrameGoTo(tk.LabelFrame):
    def __init__(self, text_button, **kwargs):
        super().__init__(**kwargs)
        # Go to
        self.button_goto = tk.Button(master=self, text=text_button)
        self.button_goto.grid(row=0, column=0, sticky="ew")
