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
    traj_paths = []

    # GUI elements: Trajectories
    header_traj = sg.Text("Provide trajectories")
    browse_project_path = sg.FolderBrowse(
        "Choose OTVision project folder", target="-input_project_path-"
    )
    input_project_path = sg.In(
        project_folder,
        size=(WIDTH_COL1, 1),
        key="-input_project_path-",
        enable_events=True,
    )
    listbox_traj_paths = sg.Listbox(
        values=traj_paths, size=(WIDTH_COL1, 20), key="-listbox_traj_paths-"
    )
    browse_traj_paths = sg.FilesBrowse(
        "Choose single trajectory files",
        key="-button_traj_paths-",
        target="-dummy_input_traj_paths-",
        enable_events=True,
    )
    dummy_input_traj_paths = sg.Input(
        key="-dummy_input_traj_paths-", enable_events=True, visible=False
    )
    text_traj_px = sg.Text(
        "No trajectory files selected.", key="-text_traj_px-", size=(WIDTH_COL1, 1)
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
        [browse_project_path],
        [input_project_path],
        [sg.Text("or")],
        [dummy_input_traj_paths, browse_traj_paths],
        [listbox_traj_paths],
        [text_traj_px],
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
        if event == "-dummy_input_traj_paths-":
            traj_paths = values["-dummy_input_traj_paths-"].split(";")
            window["-text_traj_px-"].Update(
                str(len(traj_paths)) + " trajectory files are selected."
            )
            print(traj_paths)
            window["-listbox_traj_paths-"].Update(values=traj_paths)
        elif event == "-input_project_path-":
            window["-text_traj_px-"].Update(
                "All trajectory files within the project "
                + values["-input_project_path-"]
                + " are selected."
            )

    window.close()
