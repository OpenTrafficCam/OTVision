import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
import config
from helpers.files import get_files


class FrameFiles(tk.LabelFrame):
    def __init__(self, filecategory, default_filetype, filetypes=None, **kwargs):
        super().__init__(**kwargs)
        # File type
        self.default_filetype = default_filetype
        self.filetype = self.default_filetype
        if filetypes:
            self.label_filetype = tk.Label(master=self, text="Input file type")
            self.label_filetype.grid(row=0, column=0, sticky="w")
            self.combo_filetype = ttk.Combobox(
                master=self,
                values=filetypes,
            )
            self.combo_filetype.grid(row=0, column=1, sticky="w")
            self.combo_filetype.bind("<<ComboboxSelected>>", self.set_filetype)
            self.combo_filetype.set(default_filetype)
        # File list
        self.file_list = []  # ?: Add test data?
        self.filecategory = filecategory
        self.filetypes = filetypes
        # File buttons
        self.button_add_folder = tk.Button(master=self, text="Add folder")
        self.button_add_folder.bind(
            "<Button-1>", self.add_dirs
        )  # BUG: takes 1 positional argument but 2 were given
        self.button_add_folder.grid(row=1, column=0, sticky="ew")
        self.button_add_file = tk.Button(master=self, text="Add files")
        self.button_add_file.bind(
            "<Button-1>", self.add_files
        )  # BUG: takes 1 positional argument but 2 were given
        self.button_add_file.grid(row=1, column=1, sticky="ew")
        self.button_rem_sel = tk.Button(master=self, text="Remove selected")
        self.button_rem_sel.bind("<Button-1>", self.remove_selected)
        self.button_rem_sel.grid(row=1, column=2, sticky="ew")
        self.button_rem_all = tk.Button(master=self, text="Remove all")
        self.button_rem_all.bind("<Button-1>", self.remove_all)
        self.button_rem_all.grid(row=1, column=3, sticky="ew")
        # File names
        # self.label_video = tk.Label(master=self, text="Videos to convert:")
        # self.label_video.grid(row=1, column=0, columnspan=4, sticky="w")
        self.listbox_files = tk.Listbox(master=self, width=150, selectmode="extended")
        self.listbox_files.grid(row=2, column=0, columnspan=4, sticky="ew")

    def set_filetype(self, event):
        self.filetype = self.combo_filetype.get()

    def get_listbox_files(self):
        return self.listbox_files.get(first=0, last=self.listbox_files.size() - 1)

    def get_listbox_file_indices(self):
        return self.listbox_files.get(first=0, last=self.listbox_files.size() - 1)

    def add_files(self, event):
        new_paths = list(
            filedialog.askopenfilenames(
                title=f"Select one or multiple {self.filecategory}",
                filetypes=[
                    (self.filecategory, self.filetype),
                    ("all files", "*.*"),
                ],
            )
        )
        new_paths = get_files(new_paths, self.filetype)
        self.add_to_listbox(new_paths)

    def add_dirs(self, event):
        new_dir = filedialog.askdirectory(title="Select a folder")
        new_paths = get_files(new_dir, self.filetype)
        self.add_to_listbox(new_paths)

    def add_to_listbox(self, new_paths):
        for new_path in new_paths:
            if new_path not in self.get_listbox_files():
                self.listbox_files.insert("end", new_path)

    def remove_selected(self, event):
        selection = self.listbox_files.curselection()
        self.remove_from_listbox(selection)

    def remove_all(self, event):
        selection = range(self.listbox_files.size())
        self.remove_from_listbox(selection)

    def remove_from_listbox(self, selection):
        for delta, selected_file in enumerate(selection):
            file_to_remove = selected_file - delta
            self.listbox_files.delete(first=file_to_remove)

    def debug(self, event):
        print(event)


class FrameSubmit(tk.LabelFrame):
    def __init__(self, button_label="Submit", **kwargs):
        super().__init__(**kwargs)
        # Convert
        self.button_convert = tk.Button(master=self, text=button_label)
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
