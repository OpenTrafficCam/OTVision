# OTVision: Gui classes: Frames

# Copyright (C) 2020 OpenTrafficCam Contributors
# <https://github.com/OpenTrafficCam
# <team@opentrafficcam.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import PySimpleGUI as sg
from pathlib import Path

from OTVision.config import CONFIG
from OTVision.helpers.files import get_files
from .texts import OTTextSpacer


class OTFrameFoldersFiles(sg.Frame):
    def __init__(
        self,
        default_filetype,
        filetypes=None,
        title="Step 1: Browse folders and files",
        files=[],
        width=80,
        # width=CONFIG["GUI"]["FRAMEWIDTH"],
    ):
        self.filetype = default_filetype
        if isinstance(filetypes, list) and default_filetype not in filetypes:
            self.filetypes = [default_filetype, *filetypes]
        else:
            self.filetypes = filetypes
        print(files)
        super().__init__(
            title=title,
            layout=self.get_layout(files),
            size=(width, 50),
            element_justification="center",
        )

    def drop_duplicates(self, list):
        """
        Drops duplicates from a list

        Args:
            list: list of elements of any type without duplicates

        Returns:
            cleaned_list: list of elements of any type without duplicates
        """
        cleaned_list = []
        for i in list:
            if i not in cleaned_list:
                cleaned_list.append(i)
        return cleaned_list

    def update(self, files):
        """
        Updates elements of the folders/files picker gui below

        Args:
            window: pysimplegui window element
            folders: current list of folders
            files: current list of files
            paths: current list of paths

        Returns:
            No returns
        """
        self.text_files.Update("Number of selected files: " + str(len(files)))
        self.listbox_files.Update(["(...)" + file[-80:] for file in files])

    def get_layout(self, files):

        # Constants
        WIDTH_COL1 = CONFIG["GUI"]["FRAMEWIDTH"] - 2
        # files = CONFIG["LAST FILES"]
        self.selected_files = []

        # GUI elements ADD
        if isinstance(self.filetypes, list):
            self.text_filetype = sg.Text("Filetype")
            self.drop_filetype = sg.DropDown(
                values=self.filetypes,
                default_value=self.filetype,
                key="-drop_filetype-",
                enable_events=True,
            )
        self.browse_folder = sg.FolderBrowse(
            "Add folder", key="-browse_folder-", target="-dummy_input_folder-"
        )
        self.dummy_input_folder = sg.In(
            size=(WIDTH_COL1, 1),
            key="-dummy_input_folder-",
            enable_events=True,
            visible=False,
        )
        self.browse_files = sg.FilesBrowse(
            "Add file(s)",
            key="-browse_files-",
            target="-dummy_input_files-",
            enable_events=True,
            file_types=(("", self.filetype),),
        )
        self.dummy_input_files = sg.Input(
            key="-dummy_input_files-", enable_events=True, visible=False
        )

        # GUI elements STATUS
        self.text_listbox_files = sg.Text("Selected files:")
        self.listbox_files = sg.Listbox(
            values=["(...)" + file[-80:] for file in files],
            size=(WIDTH_COL1, 8),
            key="-listbox_files-",
            enable_events=True,
            select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED,
        )
        self.text_files = sg.Text(
            "Number of selected files: " + str(len(files)),
            key="-text_files-",
            size=(WIDTH_COL1, 1),
            justification="center",
        )

        # GUI elements REMOVE
        # self.button_remove_selection = sg.Button(
        #     "Remove selection", key="-button_remove_selection-"
        # )
        # self.button_keep_selection = sg.Button(
        #     "Keep selection", key="-button_keep_selection-"
        # )
        self.button_remove_all = sg.Button("Remove all", key="-button_remove_all-")

        # Build first row conditionally
        if isinstance(self.filetypes, list):
            conditional_row = [
                self.text_filetype,
                self.drop_filetype,
                self.browse_folder,
                self.browse_files,
                self.button_remove_all,
            ]
        else:
            conditional_row = [
                self.browse_folder,
                self.browse_files,
                self.button_remove_all,
            ]

        # All the stuff inside the window
        layout = [
            [OTTextSpacer()],
            [self.dummy_input_folder, self.dummy_input_files],
            conditional_row,
            [self.text_listbox_files],
            [self.listbox_files],
            [self.text_files],
            # [
            #     self.button_keep_selection,
            #     self.button_remove_selection,
            #     self.button_remove_all,
            # ],
            [OTTextSpacer()],
        ]
        return layout

    def process_events(self, event, values, files):
        if (
            event in ["-dummy_input_folder-", "-dummy_input_files-"]
            and values[event] != ""
        ):
            files.extend(
                get_files(
                    paths=values[event].split(";"),
                    filetypes=self.filetype,
                )
            )
            files = self.drop_duplicates(files)
        elif event == "-button_remove_all-":
            files = []
        elif event == "-drop_filetype-":
            self.filetype = values[event]
        # elif event == "-listbox_files-":
        #     self.selected_files = values["-listbox_files-"]
        # elif event == "-button_keep_selection-":
        #     files = [i for i in files if i in self.selected_files]
        # elif event == "-button_remove_selection-":
        #     files = [i for i in files if i not in self.selected_files]
        self.update(files)
        return files


class OldFrameFoldersFiles(sg.Frame):
    def __init__(
        self,
        title="Step 1: Browse folders and files",
        files=[],
        width=CONFIG["GUI"]["FRAMEWIDTH"],
    ):
        super().__init__(title=title, layout=self.get_layout(files, width))
        self.ElementJustification = "center"

    def get_layout(self, files, width):  # sourcery skip
        button_browse_folders_files = sg.Button(
            "Browse files and/or folders", key="-button_browse_folders_files-"
        )
        text_files = sg.Text(
            "Number of selected files: " + str(len(files)),
            key="-text_files-",
        )
        layout = [
            [sg.Text("", size=(width, 1))],
            [button_browse_folders_files],
            [text_files],
            [sg.Text("", size=(width, 1))],
        ]
        return layout
