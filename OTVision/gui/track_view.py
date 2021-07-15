# OTVision: Python gui to track road users detected as bounding boxes
# in images or frames of videos using open source algorithms like IOU-Tracker.

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
from gui.helpers import otc_theme
from track import iou, track
from helpers.files import get_files


def main(paths=None, debug=True):

    # Initial configuration
    if debug:
        paths = CONFIG["TESTDATAFOLDER"]
        paths = get_files(
            paths=paths,
            filetypes=CONFIG["FILETYPES"]["DETECT"],
        )
    elif not paths:
        paths = CONFIG["LAST PATHS"]["DETECTIONS"]
    files = get_files(
        paths=paths,
        filetypes=CONFIG["FILETYPES"]["DETECT"],
    )

    # Get initial layout and create initial window
    layout, frame_folders_files = create_layout(files)
    window = OTSubpackageWindow(title="OTVision: Track", layout=layout)
    frame_folders_files.listbox_files.expand(expand_x=True)
    window["-progress_track-"].update_bar(0)

    # Call function to process events
    show_window = True
    while show_window:
        show_window, files = process_events(window, files, frame_folders_files)
    window.close()


def create_layout(files):

    # GUI elements: Choose Detections
    frame_folders_files = OTFrameFoldersFiles(
        default_filetype=CONFIG["FILETYPES"]["DETECT"], files=files
    )

    # GUI elements: Tracking parameters
    width_c1 = int(CONFIG["GUI"]["FRAMEWIDTH"] / 2)
    width_c2 = 10
    slider_height = 10

    # sigma_l, sigma_h, sigma_iou, t_min, t_miss_max
    text_sigma_l = sg.T(
        "Sigma l",
        justification="right",
        size=(width_c1, 1),
    )
    slider_sigma_l = sg.Slider(
        range=(0, 1),
        resolution=0.01,
        orientation="h",
        size=(width_c2, slider_height),
        default_value=CONFIG["TRACK"]["IOU"]["SIGMA_L"],
        key="-slider_sigma_l-",
    )
    text_sigma_h = sg.T(
        "Sigma h",
        justification="right",
        size=(width_c1, 1),
    )
    slider_sigma_h = sg.Slider(
        range=(0, 1),
        resolution=0.01,
        orientation="h",
        size=(width_c2, slider_height),
        default_value=CONFIG["TRACK"]["IOU"]["SIGMA_H"],
        key="-slider_sigma_h-",
    )
    text_sigma_iou = sg.T(
        "Sigma iou",
        justification="right",
        size=(width_c1, 1),
    )
    slider_sigma_iou = sg.Slider(
        range=(0, 1),
        resolution=0.01,
        orientation="h",
        size=(width_c2, slider_height),
        default_value=CONFIG["TRACK"]["IOU"]["SIGMA_IOU"],
        key="-slider_sigma_iou-",
    )
    text_t_min = sg.T(
        "t min",
        justification="right",
        size=(width_c1, 1),
    )
    slider_t_min = sg.Slider(
        range=(0, 20),  # TODO: #78 track iou t(min) boundaries for gui?
        resolution=1,
        orientation="h",
        size=(width_c2, slider_height),
        default_value=CONFIG["TRACK"]["IOU"]["T_MIN"],
        key="-slider_t_min-",
    )
    text_t_miss_max = sg.T(
        "t miss max",
        justification="right",
        size=(width_c1, 1),
    )
    slider_t_miss_max = sg.Slider(
        range=(0, 10),  # TODO: #79 track iou t(miss,max) boundaries for gui?
        resolution=1,
        orientation="h",
        size=(width_c2, slider_height),
        default_value=CONFIG["TRACK"]["IOU"]["T_MISS_MAX"],
        key="-slider_t_miss_max-",
    )
    text_overwrite = sg.T(
        "Overwrite",
        justification="right",
        size=(width_c1, 1),
    )
    check_overwrite = sg.Check(
        text="",
        default=CONFIG["TRACK"]["IOU"]["OVERWRITE"],
        key="-check_overwrite-",
    )

    frame_parameters = sg.Frame(
        "Step 2: Set parameters",
        [
            [OTTextSpacer()],
            [text_sigma_l, slider_sigma_l],
            [text_sigma_h, slider_sigma_h],
            [text_sigma_iou, slider_sigma_iou],
            [text_t_min, slider_t_min],
            [text_t_miss_max, slider_t_miss_max],
            [text_overwrite, check_overwrite],
            [OTTextSpacer()],
        ],
        size=(100, 10),
    )

    # GUI elements: Track
    button_track = sg.B("Track!", key="-button_track-")
    progress_track = sg.ProgressBar(
        max_value=len(files),
        size=(CONFIG["GUI"]["FRAMEWIDTH"] / 2, 20),
        key="-progress_track-",
    )
    frame_track = sg.Frame(
        title="Step 3: Start tracking",
        layout=[[OTTextSpacer()], [button_track], [progress_track], [OTTextSpacer()]],
        element_justification="center",
    )

    # Put layout together
    col_all = sg.Column(
        [[frame_folders_files], [frame_parameters], [frame_track]],
        scrollable=True,
        expand_y=True,
    )
    layout = [[col_all]]

    return layout, frame_folders_files


def process_events(window, files, frame_folders_files):
    """Event Loop to process "events" and get the "values" of the inputs

    Args:
        window: Window of subpackage
        files: Current file list
        frame_folders_files: Instance of gui.helpers.frames.OTFrameFoldersFiles

    Returns:
        show_window: False if window should be closed after this loop
    """

    event, values = window.read()
    print(event)

    # Close Gui
    if event in [sg.WIN_CLOSED, "Cancel", "-BUTTONBACKTOHOME-"]:
        return False, files

    # Set parameters
    elif event == "-button_track-":
        trk_config = {}
        trk_config["sigma_l"] = values["-slider_sigma_l-"]
        trk_config["sigma_h"] = values["-slider_sigma_h-"]
        trk_config["sigma_iou"] = values["-slider_sigma_iou-"]
        trk_config["t_min"] = values["-slider_t_min-"]
        trk_config["save_age"] = values["-slider_t_miss_max-"]
        for i, file in enumerate(files):
            detections, fir, filename = track.read(file)
            # ?: Which return, "new_detections" or "tracks_finished!?
            tracks_px, trajectories_geojson = track.track(
                detections=detections, trk_config=trk_config
            )
            track.write(
                tracks_px=tracks_px,
                detfile=file,
                trajectories_geojson=trajectories_geojson,
                overwrite=values["-check_overwrite-"],
            )
            window["-progress_track-"].update(current_count=i + 1, max=len(files))
        sg.popup("Job done!", title="Job done!", icon=CONFIG["GUI"]["OTC ICON"])

    # Folders and files
    files = frame_folders_files.process_events(event, values, files)
    window["-progress_track-"].update(current_count=0, max=len(files))

    return True, files
