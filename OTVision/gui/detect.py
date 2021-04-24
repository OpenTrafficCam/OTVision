# OTVision: Python gui to detect bounding boxes in images or frames of videos
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

import PySimpleGUI as sg
from config import CONFIG
from gui.helpers.frames import OTFrameFoldersFiles
from gui.helpers.windows import OTSubpackageWindow
from gui.helpers.texts import OTTextSpacer
from gui.helpers import otc_theme
from detect import yolo, detect
from helpers.files import get_files


def main(paths=None, debug=True):

    # Initial configuration
    if debug:
        paths = CONFIG["TESTDATAFOLDER"]
        paths = get_files(
            paths=paths,
            filetypes=CONFIG["CONVERT"]["OUTPUT_FILETYPE"],
        )
    elif not paths:
        paths = CONFIG["LAST PATHS"]["VIDEOS"]
    files = get_files(
        paths=paths,
        filetypes=CONFIG["FILETYPES"]["VID"],
    )
    # sg.SetOptions(font=(CONFIG["GUI"]["FONT"], CONFIG["GUI"]["FONTSIZE"]))

    # Get initial layout and create initial window
    layout, frame_folders_files = create_layout(files)
    window = OTSubpackageWindow(title="OTVision: Convert", layout=layout)
    frame_folders_files.listbox_files.expand(expand_x=True)
    window["-progress_detect-"].update_bar(0)

    # Call function to process events
    show_window = True
    while show_window:
        show_window, files = process_events(window, files, frame_folders_files)
    window.close()


def create_layout(files):

    # GUI elements: Choose videos
    frame_folders_files = OTFrameFoldersFiles(
        default_filetype=".mp4", filetypes=CONFIG["FILETYPES"]["VID"], files=files
    )

    # GUI elements: Detection weights and parameters
    width_c1 = int(CONFIG["GUI"]["FRAMEWIDTH"] / 2)
    width_c2 = 10
    slider_height = 10
    text_weights = sg.T(
        "Weights",
        justification="right",
        size=(width_c1, 1),
    )
    drop_weights = sg.DropDown(
        values=CONFIG["DETECT"]["YOLO"]["AVAILABLEWEIGHTS"],
        default_value=CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
        key="-drop_weights-",
        enable_events=True,
    )
    text_conf = sg.T(
        "Confidence",
        justification="right",
        size=(width_c1, 1),
    )
    slider_conf = sg.Slider(
        range=(0, 1),
        resolution=0.01,
        orientation="h",
        size=(width_c2, slider_height),
        default_value=CONFIG["DETECT"]["YOLO"]["CONF"],
        key="-slider_conf-",
    )
    text_iou = sg.T(
        "IOU",
        justification="right",
        size=(width_c1, 1),
    )
    slider_iou = sg.Slider(
        range=(0, 1),
        resolution=0.01,
        orientation="h",
        size=(width_c2, slider_height),
        default_value=CONFIG["DETECT"]["YOLO"]["IOU"],
        key="-slider_iou-",
    )
    text_imgsize = sg.T(
        "Image size",
        justification="right",
        size=(width_c1, 1),
    )
    slider_imgsize = sg.Slider(
        range=(500, 1000),
        resolution=10,
        orientation="h",
        size=(width_c2, slider_height),
        default_value=CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
        key="-slider_imgsize-",
    )
    text_chunksize = sg.T(
        "Chunk size",
        justification="right",
        size=(width_c1, 1),
    )
    slider_chunksize = sg.Slider(
        range=(0, 20),
        orientation="h",
        size=(width_c2, slider_height),
        default_value=CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
        key="-slider_chunksize-",
    )
    text_normalized = sg.T(
        "Normalized",
        justification="right",
        size=(width_c1, 1),
    )
    check_normalized = sg.Check(
        text="",
        default=CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
        key="-check_normalized-",
    )
    text_overwrite = sg.T(
        "Overwrite",
        justification="right",
        size=(width_c1, 1),
    )
    check_overwrite = sg.Check(
        text="",
        default=CONFIG["DETECT"]["YOLO"]["OVERWRITE"],
        key="-check_overwrite-",
    )
    frame_parameters = sg.Frame(
        "Step 2: Set parameters",
        [
            [OTTextSpacer()],
            [text_weights, drop_weights],
            [text_conf, slider_conf],
            [text_iou, slider_iou],
            [text_imgsize, slider_imgsize],
            [text_chunksize, slider_chunksize],
            [text_normalized, check_normalized],
            [text_overwrite, check_overwrite],
            [OTTextSpacer()],
        ],
        size=(100, 10),
    )

    # GUI elements: Detect
    button_detect = sg.B("Detect!", key="-button_detect-")
    progress_detect = sg.ProgressBar(
        max_value=len(files),
        size=(CONFIG["GUI"]["FRAMEWIDTH"] / 2, 20),
        key="-progress_detect-",
    )
    frame_detect = sg.Frame(
        title="Step 3: Start detection",
        layout=[[OTTextSpacer()], [button_detect], [progress_detect], [OTTextSpacer()]],
        element_justification="center",
    )

    # Put layout together
    col_all = sg.Column(
        [[frame_folders_files], [frame_parameters], [frame_detect]],
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
    elif event == "-button_detect-":
        model = yolo.loadmodel(
            weights=values["-drop_weights-"],
            conf=values["-slider_conf-"],
            iou=values["-slider_iou-"],
        )
        for i, file in enumerate(files):
            detections = yolo.detect(
                file=file,
                model=model,
                weights=values["-drop_weights-"],
                conf=values["-slider_conf-"],
                iou=values["-slider_iou-"],
                size=values["-slider_imgsize-"],
                chunksize=values["-slider_chunksize-"],
                normalized=values["-check_normalized-"],
            )
            detect.save_detections(
                detections=detections,
                infile=file,
                overwrite=values["-check_overwrite-"],
            )
            window["-progress_detect-"].update(current_count=i + 1, max=len(files))
        sg.popup("Job done!", title="Job done!", icon=CONFIG["GUI"]["OTC ICON"])

    # Folders and files
    files = frame_folders_files.process_events(event, values, files)
    window["-progress_detect-"].update(current_count=0, max=len(files))

    return True, files
