# OTVision: Python gui to transform pixel coordinates to utm coordinates
# using a set of reference points available in both pixel and utm coordinates

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
from config import CONFIG
from gui.helpers.frames import OTFrameFoldersFiles
from gui.helpers.windows import OTSubpackageWindow
from gui.helpers.texts import OTTextSpacer
from transform import transform
from helpers.files import get_files


def main(tracks_paths=None, refpts_path=None, debug=True):

    # Initial configuration
    if debug:
        tracks_paths = CONFIG["TESTDATAFOLDER"]
        tracks_paths = get_files(
            paths=tracks_paths,
            filetypes=CONFIG["FILETYPES"]["TRACK"],
        )
        refpts_path = CONFIG["TESTDATAFOLDER"]
        refpts_path = get_files(
            paths=refpts_path,
            filetypes=CONFIG["FILETYPES"]["REFPTS"],
        )
    elif not tracks_paths or not refpts_path:
        if not tracks_paths:
            tracks_paths = CONFIG["LAST PATHS"]["TRACK"]
        if not refpts_path:
            refpts_path = CONFIG["LAST PATHS"]["REFPTS"]
    tracks_files = get_files(
        paths=tracks_paths,
        filetypes=CONFIG["FILETYPES"]["TRACK"],
    )
    refpts_file = get_files(
        paths=refpts_path,
        filetypes=CONFIG["FILETYPES"]["REFPTS"],
    )

    # Get initial layout and create initial window
    layout, frame_folders_files_tracks, frame_folders_files_refpts = create_layout(
        tracks_files, refpts_file
    )
    window = OTSubpackageWindow(title="OTVision: Transform", layout=layout)
    frame_folders_files_tracks.listbox_files.expand(expand_x=True)
    frame_folders_files_refpts.listbox_files.expand(expand_x=True)
    window["-progress_transform-"].update_bar(0)

    # Call function to process events
    show_window = True
    while show_window:
        show_window, tracks_files, refpts_file = process_events(
            window,
            tracks_files,
            refpts_file,
            frame_folders_files_tracks,
            frame_folders_files_refpts,
        )
    window.close()


def create_layout(tracks_files, refpts_file):

    # GUI elements: Choose tracks
    frame_folders_files_tracks = OTFrameFoldersFiles(
        default_filetype=CONFIG["FILETYPES"]["TRACK"], files=tracks_files
    )

    # GUI elements: Choose reference points
    frame_folders_files_refpts = OTFrameFoldersFiles(
        default_filetype=CONFIG["FILETYPES"]["REFPTS"],
        files=refpts_file,
        title="Step 1: Browse reference points file",
    )

    # GUI elements: Tracking parameters
    width_c1 = int(CONFIG["GUI"]["FRAMEWIDTH"] / 2)
    width_c2 = 10
    slider_height = 10

    # GUI elements: Transform
    button_transform = sg.B("Transform!", key="-button_transform-")
    progress_transform = sg.ProgressBar(
        max_value=len(tracks_files),
        size=(CONFIG["GUI"]["FRAMEWIDTH"] / 2, 20),
        key="-progress_transform-",
    )
    frame_transform = sg.Frame(
        title="Step 3: Start transforming to UTM coordinates",
        layout=[
            [OTTextSpacer()],
            [button_transform],
            [progress_transform],
            [OTTextSpacer()],
        ],
        element_justification="center",
    )

    # Put layout together
    col_all = sg.Column(
        [
            [frame_folders_files_tracks],
            [frame_folders_files_refpts],
            [frame_transform],
        ],
        scrollable=True,
        expand_y=True,
    )
    layout = [[col_all]]

    return layout, frame_folders_files_tracks, frame_folders_files_refpts


def process_events(
    window,
    tracks_files,
    refpts_file,
    frame_folders_files_tracks,
    frame_folders_files_refpts,
):
    """Event Loop to process "events" and get the "values" of the inputs

    Args:
        window: Window of subpackage
        files: Current file list
        frame_folders_files_tracks, frame_folders_files_refpts: Instances of gui.helpers.frames.OTFrameFoldersFiles

    Returns:
        show_window: False if window should be closed after this loop
    """

    event, values = window.read()
    print(event)

    # Close Gui
    if event in [sg.WIN_CLOSED, "Cancel", "-BUTTONBACKTOHOME-"]:
        return False, tracks_files, refpts_file

    # Set parameters
    elif event == "-button_transform-":
        tracks_utm = transform.main(tracks_files=tracks_files, refpts_file=refpts_file)
        window["-progress_transform-"].update(
            current_count=i + 1, max=len(tracks_files)
        )
        sg.popup("Job done!", title="Job done!", icon=CONFIG["GUI"]["OTC ICON"])

    # Folders and files
    tracks_files = frame_folders_files_tracks.process_events(
        event, values, tracks_files
    )
    refpts_file = frame_folders_files_refpts.process_events(event, values, refpts_file)
    window["-progress_transform-"].update(current_count=0, max=len(tracks_files))

    return True, tracks_files, refpts_file
