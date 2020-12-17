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


from tkinter.font import BOLD
import PySimpleGUI as sg
from gui import browse_folders_and_files
from gui.sg_otc_theme import (
    OTC_ICON,
    OTC_BUTTON,
    OTC_THEME,
    OTC_FONT,
    OTC_FONTSIZE,
)
import cv2
import time


# Constants
WIDTH_COL1 = 150
sg.SetOptions(font=(OTC_FONT, OTC_FONTSIZE))


def create_window(OTC_ICON, layout):
    window_title = "Transform trajectories from pixel to world coordinates"
    window = (
        sg.Window(window_title, icon=OTC_ICON, resizable=True)
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
    return window


def main(sg_theme=OTC_THEME):
    # Lists
    # traj_paths = []
    traj_folders = []
    traj_files = []

    # GUI elements: Trajectories
    header_traj = sg.Text("Step 1: Provide trajectories")
    button_browse_traj = sg.Button(
        "Browse trajectory files", key="-button_browse_traj-"
    )
    text_traj_folders = sg.Text(
        "Number of selected folders: " + str(len(traj_folders)),
        key="-text_traj_folders-",
        size=(WIDTH_COL1, 1),
    )
    text_traj_files = sg.Text(
        "Number of selected files: " + str(len(traj_files)),
        key="-text_traj_files-",
        size=(WIDTH_COL1, 1),
    )

    # GUI elemnts: Reference points
    header_refpts = sg.Text(
        "Step 2: Provide reference points in both pixel and world coordinates",
    )
    input_refpts = sg.In(
        key="-input_refpts-", size=(WIDTH_COL1, 1), enable_events=True, visible=True,
    )
    browse_refpts = sg.FileBrowse(
        "Choose existing reference points",
        key="-browse_refpts-",
        target="-input_refpts-",
        enable_events=True,
    )
    filesaveas_refpts = sg.FileSaveAs(
        "Save reference points as ...", key="-filesaveas_refpts-"
    )

    # GUI elements: transform to world
    header_transform = sg.Text("Step 3: Start transformation to world coordinates")

    # Video properties
    video = cv2.VideoCapture(
        r"C:\Users\Baerwolff\Desktop\Lenovo_Arbeit\2020-02-20_Validierungsmessung_Radeberg\raspberrypi_FR20_2020-02-20_11-30-00.flv"
    )
    video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    video_fps = video.get(cv2.CAP_PROP_FPS)
    video_total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    a_id = None
    play_video = False
    ret = False
    video_curr_frame = 0

    # Gui elements: Video player
    callback_image = sg.Graph(
        (video_width, video_height),
        (0, video_height),
        (video_width, 0),
        key="-callback_image-",
        enable_events=True,
        drag_submits=True,
        background_color="black",
    )
    button_play = sg.Button("Play", key="-button_play-")
    button_pause = sg.Button("Pause", key="-button_pause-")
    slider_video = sg.Slider(
        range=(0, video_total_frames),
        size=(60, 10),
        orientation="h",
        key="-slider_video-",
    )

    # GUI elements: Exit gui data
    button_back_to_otvision = sg.Button(
        "", key="-button_back_to_otvision-", image_data=OTC_BUTTON, border_width=0,
    )

    # All the stuff inside the window
    layout = [
        [header_traj],
        [button_browse_traj],
        [text_traj_folders],
        [text_traj_files],
        [sg.Text("")],
        [header_refpts],
        [browse_refpts, input_refpts],
        [callback_image],
        [button_play, button_pause],
        [slider_video],
        [filesaveas_refpts],
        [sg.Text("")],
        [header_transform],
        [sg.Text("")],
        [button_back_to_otvision],
    ]

    # Create the Window
    window = create_window(OTC_ICON, layout)

    # Event Loop to process "events" and get the "values" of the inputs
    while video.isOpened():
        event, values = window.read(timeout=0)
        window["-column-"].expand(True, True)
        # screen_width, screen_height = window.get_screen_size()
        if (
            event == sg.WIN_CLOSED
            or event == "Cancel"
            or event == "-button_back_to_otvision-"
        ):  # if user closes window or clicks cancel
            break
        # Video stuff
        if event == "-button_play-":
            play_video = True
        elif event == "-button_pause-" or not ret:
            play_video = False
        if play_video:
            ret, frame = video.read()
            imgbytes = cv2.imencode(".png", frame)[1].tobytes()
            if a_id:
                window["-callback_image-"].delete_figure(a_id)  # delete previous image
            a_id = window["-callback_image-"].draw_image(
                data=imgbytes, location=(0, 0)
            )  # draw new image
            window["-callback_image-"].TKCanvas.tag_lower(a_id)  # move image to bottom
            time.sleep(1 / video_fps)
            # if someone moved the slider manually, the jump to that frame
            if int(values["-slider_video-"]) != video_curr_frame - 1:
                video_curr_frame = int(values["-slider_video-"])
                video.set(cv2.CAP_PROP_POS_FRAMES, video_curr_frame)
            window["-slider_video-"].update(video_curr_frame)
            # video_curr_frame += 1
        # Video callbacks
        if event == "-callback_image-":
            refpts_px_values = values["-callback_image-"]
            window["-callback_image-"].draw_circle(
                refpts_px_values, 5, fill_color="red", line_color="red"
            )
        # Other gui stuff
        if event == "-button_browse_traj-":
            traj_folders, traj_files = browse_folders_and_files.main(
                title="Select trajectories",
                filetype="_trackspx.json",
                input_folders=traj_folders,
                input_files=traj_files,
            )
            window["-text_traj_folders-"].Update(
                "Number of selected folders: " + str(len(traj_folders))
            )
            window["-text_traj_files-"].Update(
                "Number of selected trajectory files: " + str(len(traj_files))
            )
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


# To Dos:
# - Rearange loop to enable Play, Pause and Slider simultaneously
# - Load video automatically by file name of "[...]_trackspx.json"
#   or otherwise by browse button
# - Remove error when video runs out of frames
# - Draw on every mouse click (everey now and then it doesnt work)
# - Draw crosshair instead of circle
# - Add embededding or popup for map
# - Append clicked pixel and utm refpts to dictionary
# - Button to load/save this dictionary from/to json/pkl/npy/txt
# - Button to call transfrom_to_world function passing refpts
# - Add function for accuracy evaluation to transform_to_world
# - Some element to show results of accuracy evaluation
