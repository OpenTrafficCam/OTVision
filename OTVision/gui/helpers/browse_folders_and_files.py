# OTVision: Python module to calculate homography matrix from reference
# points and transform trajectory points from pixel into world coordinates.

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
from PySimpleGUI.PySimpleGUI import LISTBOX_SELECT_MODE_EXTENDED
from gui.helpers.sg_otc_theme import OTC_ICON, OTC_THEME, OTC_FONT


# Constants
sg.SetOptions(font=OTC_FONT)


def drop_duplicates(list):
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


def update_gui(window, folders, files, paths):
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
    window["-text_folders-"].Update("Number of selected folders: " + str(len(folders)))
    window["-text_files-"].Update("Number of selected files: " + str(len(files)))
    window["-listbox_paths-"].Update(paths)


def format_filetypes(filetypes):
    """
    check if _filetypes_ is list of str and transform it

    Args:
        filetypes: list of file extension with or without leading "." or "_"

    Returns:
        filetypes: file extension with leading "." (except leading "_" was passed)
    """
    if type(filetypes) is not list:
        filetypes = [filetypes]
    for filetype in filetypes:
        if type(filetype) is str:
            if not filetype.startswith("_"):
                if not filetype.startswith("."):
                    filetype = "." + filetype
                filetype = filetype.lower()
        else:
            raise TypeError("filetype needs to be a str")
    return filetypes


def main(
    title="Select folders/files",
    filetypes=[],
    input_folders=[],
    input_files=[],
    sg_theme=OTC_THEME,
):
    """
    Window for picking folders/files for further processing with OTVision tools

    Args:
        title: title of the window (default: "Select folders/files")
        filetypes ([str]): ending of the files to select.
        (Preceding "_" prevents adding a '.'; default: All files)
        input_folders: list of pre-selected folders (default: empty list)
        input_files: list of pre-selected files (default: empty list)

    Returns:
        folders: list of selected folders
        files: list of selected files
    """
    filetypes = format_filetypes(filetypes)

    # Constants
    WIDTH_COL1 = 150

    # Create new lists of files, folders and paths
    # ("new list = old list" would not copy old list,
    # instead just changes reference to the list)
    folders = []
    files = []
    paths = []
    selected_paths = []
    folders.extend(input_folders)
    files.extend(input_files)
    paths.extend(folders)
    paths.extend(files)

    # GUI elements ADD
    browse_folder = sg.FolderBrowse(
        "Add folder",
        key="-browse_folder-",
        target="-dummy_input_folder-",
    )
    dummy_input_folder = sg.In(
        size=(WIDTH_COL1, 1),
        key="-dummy_input_folder-",
        enable_events=True,
        visible=False,
    )
    browse_files = sg.FilesBrowse(
        "Add file(s)",
        key="-browse_files-",
        target="-dummy_input_files-",
        enable_events=True,
        file_types=(("", filetypes),),
    )
    dummy_input_files = sg.Input(
        key="-dummy_input_files-", enable_events=True, visible=False
    )

    # GUI elements STATUS
    text_listbox_paths = sg.Text("Selected folders and files:")
    listbox_paths = sg.Listbox(
        values=paths,
        size=(WIDTH_COL1, 20),
        key="-listbox_paths-",
        enable_events=True,
        select_mode=LISTBOX_SELECT_MODE_EXTENDED,
    )
    text_folders = sg.Text(
        "Number of selected folders: " + str(len(folders)),
        key="-text_folders-",
        size=(WIDTH_COL1, 1),
    )
    text_files = sg.Text(
        "Number of selected files: " + str(len(files)),
        key="-text_files-",
        size=(WIDTH_COL1, 1),
    )

    # GUI elements REMOVE
    button_remove_selection = sg.Button(
        "Remove selection", key="-button_remove_selection-"
    )
    button_keep_selection = sg.Button("Keep selection", key="-button_keep_selection-")
    button_remove_all = sg.Button("Remove all", key="-button_remove_all-")

    # GUI elements QUIT
    button_ok = sg.Button("Ok", key="-button_ok-")
    button_cancel = sg.Button("Cancel", key="-button_cancel-")

    # All the stuff inside the window
    layout = [
        [dummy_input_folder, dummy_input_files],
        [browse_folder, browse_files],
        [text_listbox_paths],
        [listbox_paths],
        [text_folders],
        [text_files],
        [button_keep_selection, button_remove_selection, button_remove_all],
        [],
        [button_ok, button_cancel],
    ]

    # Create the Window
    window = sg.Window(title, layout, resizable=True, icon=OTC_ICON).Finalize()

    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "-button_cancel-":
            window.close()
            return input_folders, input_files
        elif event == "-button_ok-":
            window.close()
            return folders, files
        elif event == "-dummy_input_folder-" and values["-dummy_input_folder-"] != "":
            folders.append(values["-dummy_input_folder-"])
            window["-dummy_input_folder-"].Update("")
            folders = drop_duplicates(folders)
            paths.extend(folders)
            paths = drop_duplicates(paths)
            update_gui(window, folders, files, paths)
        elif event == "-dummy_input_files-" and values["-dummy_input_files-"] != "":
            files.extend(values["-dummy_input_files-"].split(";"))
            window["-dummy_input_files-"].Update("")
            files = drop_duplicates(files)
            paths.extend(files)
            paths = drop_duplicates(paths)
            update_gui(window, folders, files, paths)
        elif event == "-button_remove_all-":
            folders = []
            files = []
            paths = []
            update_gui(window, folders, files, paths)
        elif event == "-listbox_paths-":
            selected_paths = values["-listbox_paths-"]
        elif event == "-button_keep_selection-":
            folders = [i for i in folders if i in selected_paths]
            files = [i for i in files if i in selected_paths]
            paths = selected_paths
            window["-text_folders-"].Update(str(len(folders)) + " folders selected.")
            window["-text_files-"].Update(str(len(files)) + " files selected.")
            update_gui(window, folders, files, paths)
        elif event == "-button_remove_selection-":
            folders = [i for i in folders if i not in selected_paths]
            files = [i for i in files if i not in selected_paths]
            paths = [i for i in paths if i not in selected_paths]
            update_gui(window, folders, files, paths)


if __name__ == "__main__":
    print(
        main(
            title="Select images",
            filetypes="png",
            input_folders=["Input_Test_Folder"],
            input_files=["Input_Test_File.ending"],
        )
    )
