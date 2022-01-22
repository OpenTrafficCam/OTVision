import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

from numpy import var
from config import CONFIG, PAD
from helpers.files import get_files
from pathlib import Path


class FrameFileTree(tk.LabelFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # File dict
        self.files_dict = {}
        self.vid_filetype = CONFIG["DEFAULT_FILETYPE"]["VID"].replace(".", "")

        # Frame for controls
        self.frame_controls = tk.Frame(master=self)
        self.frame_controls.pack()

        # File type dropdowns
        self.label_vid_filetype = tk.Label(
            master=self.frame_controls, text="Video file type"
        )
        self.label_vid_filetype.grid(row=0, column=4, sticky="w")
        self.combo_vid_filetype = ttk.Combobox(
            master=self.frame_controls,
            values=[str.replace(".", "") for str in CONFIG["FILETYPES"]["VID"]],
            width=5,
        )
        self.combo_vid_filetype.grid(row=0, column=5, sticky="w")
        self.combo_vid_filetype.bind("<<ComboboxSelected>>", self.set_vid_filetype)
        self.combo_vid_filetype.set(self.vid_filetype)

        # File buttons
        self.button_add_folder = tk.Button(
            master=self.frame_controls, text="Add folder"
        )
        self.button_add_folder.bind("<ButtonRelease-1>", self.add_dirs)
        self.button_add_folder.grid(row=0, column=0, sticky="ew")
        self.button_add_file = tk.Button(master=self.frame_controls, text="Add files")
        self.button_add_file.bind("<ButtonRelease-1>", self.add_files)
        self.button_add_file.grid(row=0, column=1, sticky="ew")
        self.button_rem_sel = tk.Button(
            master=self.frame_controls, text="Remove selected"
        )
        self.button_rem_sel.bind("<ButtonRelease-1>", self.remove_selected)
        self.button_rem_sel.grid(row=0, column=2, sticky="ew")
        self.button_rem_all = tk.Button(master=self.frame_controls, text="Remove all")
        self.button_rem_all.bind("<ButtonRelease-1>", self.remove_all)
        self.button_rem_all.grid(row=0, column=3, sticky="ew")

        # Frame for treeview
        self.frame_tree = tk.Frame(master=self)
        self.frame_tree.pack(**PAD, fill="x")  # BUG: Treeview fills only to the left

        # Files treeview
        self.tree_files = ttk.Treeview(master=self.frame_tree)
        self.tree_files.bind("<ButtonRelease-3>", self.deselect_tree_files)
        tree_files_cols = {
            "#0": "File",
            "h264": "h264",
            "video": self.vid_filetype,
            "otdet": "otdet",
            "ottrk": "ottrk",
        }
        self.tree_files["columns"] = tuple(
            {k: v for k, v in tree_files_cols.items() if k != "#0"}.keys()
        )
        for tree_files_col_id, tree_files_col_text in tree_files_cols.items():
            if tree_files_col_id == "#0":
                anchor = "w"
                width = 700
            else:
                anchor = "center"
                width = 50
            self.tree_files.column(tree_files_col_id, width=width, anchor=anchor)
            self.tree_files.heading(
                tree_files_col_id, text=tree_files_col_text, anchor=anchor
            )
        self.tree_files.pack(side="left")

        # Scrollbar for treeview
        self.tree_scrollbar = ttk.Scrollbar(
            master=self.frame_tree, orient="vertical", command=self.tree_files.yview
        )
        self.tree_scrollbar.pack(side="right", fill="y")
        self.tree_files.configure(yscrollcommand=self.tree_scrollbar.set)

    def set_vid_filetype(self, event):
        self.vid_filetype = self.combo_vid_filetype.get()
        for path in self.files_dict.keys():
            self.update_files_dict_values(path)
        self.tree_files.heading("video", text=self.vid_filetype)
        self.update_tree_files()

    def add_dirs(self, event):
        new_dir = filedialog.askdirectory(title="Select a folder")
        new_paths = get_files(new_dir, self.filetype)
        self.add_to_listbox(new_paths)

    def add_files(self, event):
        # Show filedialog
        new_paths = list(
            filedialog.askopenfilenames(
                title="Select one or multiple files",
                filetypes=[
                    (".h264", ".h264"),
                    (".mp4", ".mp4"),
                    (".otdet", ".otdet"),
                    (".ottrk", ".ottrk"),
                    ("all files", "*.*"),
                ],
            )
        )
        # Check paths
        new_paths = get_files(new_paths)
        self.add_to_files_dict(new_paths)

    def remove_selected(self, event):
        for item in self.tree_files.selection():
            del self.files_dict[self.tree_files.item(item)["text"]]
        self.update_tree_files()

    def remove_all(self, event):
        self.files_dict = {}
        self.update_tree_files()

    def add_to_files_dict(self, paths):
        for path in paths:
            self.files_dict[path] = {}
            self.update_files_dict_values(path)
        self.update_tree_files()

    def update_files_dict(self):
        for path in self.files_dict.keys():
            self.update_files_dict_values(path)
        self.update_tree_files()

    def update_files_dict_values(self, path):
        TRUE_SYMBOL = "\u2705"  # "\u2713"  # "\u2714"
        FALSE_SYMBOL = "\u274E"  # "\u2717"  # "\u2718"
        self.files_dict[path]["filename"] = Path(path).stem
        self.files_dict[path]["h264"] = (
            TRUE_SYMBOL if Path(path).with_suffix(".h264").is_file() else FALSE_SYMBOL
        )
        self.files_dict[path]["video"] = (
            TRUE_SYMBOL
            if Path(path).with_suffix(f".{self.vid_filetype}").is_file()
            else FALSE_SYMBOL
        )
        self.files_dict[path]["otdet"] = (
            TRUE_SYMBOL if Path(path).with_suffix(".otdet").is_file() else FALSE_SYMBOL
        )
        self.files_dict[path]["ottrk"] = (
            TRUE_SYMBOL if Path(path).with_suffix(".ottrk").is_file() else FALSE_SYMBOL
        )

    def update_tree_files(self):
        self.tree_files.delete(*self.tree_files.get_children())
        for path, file_values in self.files_dict.items():
            self.tree_files.insert(
                parent="",
                index="end",
                text=path,
                values=(
                    file_values["h264"],
                    file_values["video"],
                    file_values["otdet"],
                    file_values["ottrk"],
                ),
            )

    def get_tree_files(self):
        return [
            self.tree_files.item(item)["text"]
            for item in self.tree_files.get_children()
        ]
        # return [self.tree_files.set(item, 0) for item in self.tree_files.get_children()]

    def deselect_tree_files(self, events):
        for item in self.tree_files.selection():
            self.tree_files.selection_remove(item)


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
        self.button_add_folder.bind("<ButtonRelease-1>", self.add_dirs)
        self.button_add_folder.grid(row=1, column=0, sticky="ew")
        self.button_add_file = tk.Button(master=self, text="Add files")
        self.button_add_file.bind("<ButtonRelease-1>", self.add_files)
        self.button_add_file.grid(row=1, column=1, sticky="ew")
        self.button_rem_sel = tk.Button(master=self, text="Remove selected")
        self.button_rem_sel.bind("<ButtonRelease-1>", self.remove_selected)
        self.button_rem_sel.grid(row=1, column=2, sticky="ew")
        self.button_rem_all = tk.Button(master=self, text="Remove all")
        self.button_rem_all.bind("<ButtonRelease-1>", self.remove_all)
        self.button_rem_all.grid(row=1, column=3, sticky="ew")
        # File list
        self.listbox_files = tk.Listbox(master=self, width=150, selectmode="extended")
        self.listbox_files.yview()
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


class FrameRun(tk.Frame):
    def __init__(self, button_label="Run!", **kwargs):
        super().__init__(**kwargs)
        # Run
        self.button_run = tk.Button(master=self, text=button_label)
        self.button_run.pack(fill="both")
        # Include in chained run
        self.checkbutton_run_chained_var = tk.BooleanVar()
        self.checkbutton_run_chained = tk.Checkbutton(
            master=self,
            text="Include in chained run",
            variable=self.checkbutton_run_chained_var,
        )
        self.checkbutton_run_chained.pack()
        # self.checkbutton_run_chained.select()
        # # Progress bar  # TODO
        # self.progress = ttk.Progressbar(master=self)
        # self.progress.grid(row=1, column=0, sticky="ew")


class FrameRunChained(tk.LabelFrame):
    def __init__(self, button_label="Run chained!", **kwargs):
        super().__init__(**kwargs)
        # Run
        self.button_run = tk.Button(master=self, text=button_label)
        self.button_run.pack(**PAD, fill="x", expand=1)
        self.button_run.bind("<ButtonRelease-1>", self.run)
        # # Progress bar  # TODO
        # self.progress = ttk.Progressbar(master=self)
        # self.progress.grid(row=1, column=0, sticky="ew")

    def run(self, event):
        if self.master.frame_convert.frame_run.checkbutton_run_chained_var.get():
            self.master.frame_convert.frame_run.run(event)
        if self.master.frame_detect.frame_run.checkbutton_run_chained_var.get():
            self.master.frame_detect.frame_run.run(event)
        if self.master.frame_track.frame_run.checkbutton_run_chained_var.get():
            self.master.frame_track.frame_run.run(event)
