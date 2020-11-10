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


def main(project_folder=""):
    # Constants
    WIDTH_COL1 = 150

    # Lists
    # traj_paths = []
    traj_folders = []
    traj_files = []

    # GUI elements: Trajectories
    header_traj = sg.Text("Provide trajectories")
    browse_traj_folder = sg.FolderBrowse(
        "Add folder with trajectory files",
        key="-browse_traj_folder-",
        target="-dummy_input_traj_folder-",
    )
    dummy_input_traj_folder = sg.In(
        size=(WIDTH_COL1, 1),
        key="-dummy_input_traj_folder-",
        enable_events=True,
        visible=False,
    )
    """listbox_traj_paths = sg.Listbox(
        values=traj_paths, size=(WIDTH_COL1, 20), key="-listbox_traj_paths-"
    )
    """
    browse_traj_files = sg.FilesBrowse(
        "Add single trajectory files",
        key="-browse_traj_files-",
        target="-dummy_input_traj_files-",
        enable_events=True,
    )
    dummy_input_traj_files = sg.Input(
        key="-dummy_input_traj_files-", enable_events=True, visible=False
    )
    text_traj_folders = sg.Text(
        "0 folders selected.", key="-text_traj_folders-", size=(WIDTH_COL1, 1),
    )
    text_traj_files = sg.Text(
        "0 files selected.", key="-text_traj_files-", size=(WIDTH_COL1, 1),
    )
    button_show_traj_selection = sg.Button(
        "Show selection", key="-button_show_traj_selection-"
    )
    button_clear_traj_selection = sg.Button(
        "Clear selection", key="-button_clear_traj_selection-"
    )

    # GUI elemnts: Reference points
    header_refpts = sg.Text(
        "Provide reference points in both pixel and world coordinates"
    )
    button_click_refpts = sg.Button(
        "Click new reference points", key="-button_click_refpts-"
    )
    dummy_input_refpts_path = sg.In(
        key="-dummy_input_refpts_path-", enable_events=True, visible=False
    )
    browse_refpts_path = sg.FileBrowse(
        "Choose existing reference points",
        key="-browse_refpts_path-",
        target="-dummy_input_refpts_path-",
        enable_events=True,
    )

    # GUI elements: Exit gui data
    button_back_to_otvision = sg.Button(
        "Back to OTVision", key="-button_back_to_otvision-"
    )

    # All the stuff inside the window
    layout = [
        [header_traj],
        [
            browse_traj_folder,
            browse_traj_files,
            button_show_traj_selection,
            button_clear_traj_selection,
        ],
        [dummy_input_traj_folder, dummy_input_traj_files],
        [text_traj_folders],
        [text_traj_files],
        [header_refpts],
        [button_click_refpts, dummy_input_refpts_path, browse_refpts_path],
        [button_back_to_otvision],
    ]

    # Create the Window
    window_title = "Transform trajectories from pixel to world coordinates"
    window = sg.Window(window_title, layout).Finalize()

    # Make window fullscreen
    window.Maximize()

    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if (
            event == sg.WIN_CLOSED
            or event == "Cancel"
            or event == "-button_back_to_otvision-"
        ):  # if user closes window or clicks cancel
            break
        elif event == "-dummy_input_traj_folder-":
            traj_folders.append(values["-dummy_input_traj_folder-"])
            print("traj_folders" + str(traj_folders))
            if len(traj_folders) == 1:
                text_traj_folders_label = " folder selected."
            else:
                text_traj_folders_label = " folders selected."
            window["-text_traj_folders-"].Update(
                str(len(traj_folders)) + text_traj_folders_label
            )
        elif event == "-dummy_input_traj_files-":
            traj_files.extend(values["-dummy_input_traj_files-"].split(";"))
            print("traj_files: " + str(traj_files))
            if len(traj_files) == 1:
                text_traj_files_label = " file selected."
            else:
                text_traj_files_label = " files selected."
            window["-text_traj_files-"].Update(
                str(len(traj_files)) + text_traj_files_label
            )
        elif event == "-button_clear_selection-":
            traj_folders = []
            traj_files = []
            window["-text_traj_folders-"].Update("0 folders selected.")
            window["-text_traj_files-"].Update("0 files selected.")

    window.close()

# To Dos
# - Code "clear selection" button, which lists traj_folders and traj_files
#   and updates text
# - Remove duplicates from lists traj_folders and traj_files instantly after browsing