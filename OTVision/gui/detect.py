# OTVision: Python module to detect bounding boxes in images or frames of videos
# using deep learning algorithms like YOLOv5.

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

from PySimpleGUI.PySimpleGUI import DEFAULT_TEXT_JUSTIFICATION, Text
from gui import browse_folders_and_files
from gui.sg_otc_theme import (
    OTC_ICON,
    OTC_BUTTON,
    OTC_THEME,
    OTC_FONT,
    OTC_FONTSIZE,
)
import config
from detect.detect import main as detect

import PySimpleGUI as sg
from pathlib import Path


# Constants
WIDTH_COL1 = 150
sg.SetOptions(font=(OTC_FONT, OTC_FONTSIZE))
otvision_user_settings = config.read_user_settings()
try:
    LAST_VIDEO_PATH = otvision_user_settings["PATHS"]["LAST_VIDEO_PATH"]
except KeyError:
    LAST_VIDEO_PATH = ""


def main(sg_theme=OTC_THEME):
    folders = []
    files = []
    total_files = []
    filetypes = [".mov", ".avi", ".mp4", ".mpg", ".mpeg", ".m4v", ".wmv", ".mkv"]

    # Get initial layout and create initial window
    layout, text_status_detect = create_layout(folders, files, total_files)
    window = sg.Window(
        title="OTVision: detect",
        layout=layout,
        icon=OTC_ICON,
        location=(0, 0),
        resizable=True,
        finalize=True,
    )
    # window = create_window(
    #     OTC_ICON,
    #     create_layout(folders, files, total_files),
    # )

    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read(timeout=0)
        # Close Gui
        if (
            event == sg.WIN_CLOSED
            or event == "Cancel"
            or event == "-button_back_to_otvision-"
        ):  # if user closes window or clicks cancel
            break
        # Folders and files
        elif event == "-button_browse_folders_files-":
            folders, files = browse_folders_and_files.main(
                title="Select videos",
                filetypes=filetypes,
                input_folders=folders,
                input_files=files,
            )
            window["-text_folders-"].Update(
                "Number of selected folders: " + str(len(folders))
            )
            window["-text_files-"].Update(
                "Number of selected files: " + str(len(files))
            )
        # Detector parameterization
        # TODOs: get_otvision_defaults, get_user_defaults, set_user_defaults
        # Detection
        elif event == "-button_detect-":
            paths = folders + files
            if len(paths) > 0:
                text_status_detect.Update("Detection is running!")
                det_config = {
                    "weights": values["-optionmenu_weights-"],
                    "conf": values["-slider_conf-"],
                    "iou": values["-slider_iou-"],
                    "size": int(values["-slider_size-"]),
                    "chunksize": int(values["-slider_chunksize-"]),
                    "normalized": values["-checkbox_normalized-"],
                }
                detect(paths, filetypes, det_config)
                text_status_detect.Update("Detection was successful!")
            else:
                text_status_detect.Update("No folders or files to detect!")

    window.close()


def create_layout(folders, files, total_files):
    # GUI elements: Choose videos
    size_col1 = (20, 5)
    size_col2 = (10, 5)
    button_browse_folders_files = sg.Button(
        "Browse files and/or folders", key="-button_browse_folders_files-"
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
    frame_folders_files = sg.Frame(
        "Step 1: Choose videos or images",
        layout=[
            [button_browse_folders_files],
            [text_folders],
            [text_files],
        ],
    )

    # GUI elements: Detection model and parameters
    size_col = 20
    size_row1 = (size_col, 1)
    size_row2 = (size_col, 10)
    optionmenu_weights = sg.OptionMenu(
        values=("yolov5s", "yolov5m", "yolov5l", "yolov5x"),
        default_value="yolov5x",
        size=size_row2,
        key="-optionmenu_weights-",
    )
    slider_conf = sg.Slider(
        range=(0, 1),
        resolution=0.01,
        orientation="h",
        size=size_row2,
        default_value=0.25,
        key="-slider_conf-",
    )
    slider_iou = sg.Slider(
        range=(0, 1),
        resolution=0.01,
        orientation="h",
        size=size_row2,
        default_value=0.45,
        key="-slider_iou-",
    )
    slider_size = sg.Slider(
        range=(500, 1000),
        orientation="h",
        size=size_row2,
        default_value=640,
        key="-slider_size-",
    )
    slider_chunksize = sg.Slider(
        range=(0, 20),
        orientation="h",
        size=size_row2,
        default_value=0,
        key="-slider_chunksize-",
    )
    checkbox_normalized = sg.Checkbox(
        text="", size=(10, 1), default=False, key="-checkbox_normalized-"
    )
    frame_det_config = sg.Frame(
        "Step 2: Parametrize detector",
        layout=[
            [
                sg.Frame("Weights", [[optionmenu_weights]]),
                sg.Frame("Confidence", [[slider_conf]]),
                sg.Frame("IOU", [[slider_iou]]),
                sg.Frame("Size", [[slider_size]]),
                sg.Frame("Chunksize", [[slider_chunksize]]),
                sg.Frame("Normalized", [[checkbox_normalized]]),
            ]
        ],
    )

    # GUI elements: Detect
    size_col1 = (30, 1)
    size_col2 = (20, 5)
    button_detect = sg.Button("Detect", size=size_col1, key="-button_detect-")
    text_status_detect = sg.Text(
        "Hit the button to start detection!",
        enable_events=True,
        key="-text_status_detect-",
    )
    progressbar_detect = sg.ProgressBar(
        max_value=total_files,
        orientation="h",
        size=size_col1,
        key="-progressbar_detect-",
    )
    frame_detect = sg.Frame(
        title="Step 3: Start detection", layout=[[button_detect], [text_status_detect]]
    )

    # GUI elements: Detections
    # TODO

    # Put GUI elemnts in a layout
    layout = [
        [frame_folders_files],
        [frame_det_config],
        [frame_detect],
    ]

    return layout, text_status_detect


def create_window(OTC_ICON, layout, window_location=(0, 0), window_size=None):
    window_title = "OTVision: Detect"
    window = (
        sg.Window(window_title, icon=OTC_ICON, resizable=True, location=window_location)
        .Layout(
            [
                [
                    sg.Column(
                        layout=layout,
                        key="-column-",
                        scrollable=True,
                        vertical_scroll_only=False,
                        expand_x=True,
                        expand_y=True,
                    )
                ]
            ]
        )
        .Finalize()
    )
    if window_size is None:
        window.Maximize()
    else:
        window.Size = window_size
    return window


def process_events():
    pass
