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
from gui import browse_folders_and_files
from gui.helpers.sg_otc_theme import (
    OTC_ICON,
    OTC_BUTTON,
    OTC_THEME,
    OTC_FONT,
    OTC_FONTSIZE,
)
import cv2
import datetime
import pause
import config
import os


# Constants
WIDTH_COL1 = 150
sg.SetOptions(font=(OTC_FONT, OTC_FONTSIZE))
PLAYER_FPS = 10
otvision_user_settings = config.read_user_settings()
try:
    LAST_VIDEO_PATH = otvision_user_settings["PATHS"]["LAST_VIDEO_PATH"]
except KeyError:
    LAST_VIDEO_PATH = ""


def log_videoplayer(pos, video, values):
    print(
        pos
        + " Video: "
        + str(int(video.get(cv2.CAP_PROP_POS_FRAMES)))
        + " vs. Slider: "
        + str(int(values["-slider_video-"]) + 1)
    )


def now_msec():
    return datetime.datetime.timestamp(datetime.datetime.now())


def create_layout(graph_video, slider_video, traj_folders, traj_files):
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

    # Gui elements: Video player
    input_video = sg.In(
        LAST_VIDEO_PATH,
        key="-input_video-",
        size=(WIDTH_COL1, 1),
        enable_events=True,
        visible=True,
    )
    browse_video = sg.FileBrowse(
        "Choose video",
        key="-browse_video-",
        target="-input_video-",
        enable_events=True,
    )
    button_play = sg.Button("Play", key="-button_play-")
    button_pause = sg.Button("Pause", key="-button_pause-")
    text_speed = sg.Text("Speed x ")
    spin_speed = sg.Spin(
        values=["0.01", "0.1", "0.25", "0.5", "1", "2", "5", "10"],
        initial_value="1",
        key="-spin_speed-",
        size=(5, 1),
        enable_events=True,
    )
    text_video_time = sg.Text("--:--")

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
        [browse_video, input_video],
        [graph_video],
        [button_play, button_pause, text_speed, spin_speed, text_video_time],
        [slider_video],
        [filesaveas_refpts],
        [sg.Text("")],
        [header_transform],
        [sg.Text("")],
        [button_back_to_otvision],
    ]
    return layout


def create_window(OTC_ICON, layout, window_location=(0, 0), window_size=None):
    window_title = "Transform trajectories from pixel to world coordinates"
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


def create_graph_video(width=640, height=480):
    graph_video = sg.Graph(
        canvas_size=(width, height),
        graph_bottom_left=(0, height),
        graph_top_right=(width, 0),
        key="-graph_video-",
        enable_events=True,
        drag_submits=True,
        background_color="black",
    )
    return graph_video


def create_slider_video(video_total_frames=0):
    slider_video = sg.Slider(
        range=(0, video_total_frames),
        size=(60, 10),
        orientation="h",
        key="-slider_video-",
    )
    return slider_video


def prepare_video(path, window, traj_folders, traj_files):

    # Get video and properties with opencv
    video = cv2.VideoCapture(path)
    video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    video_fps = video.get(cv2.CAP_PROP_FPS)
    video_total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    a_id = None
    play_video = False
    ret = False
    ret, frame = video.read()

    # Recreate pysimplegui window
    graph_video = create_graph_video(400, 300)
    slider_video = create_slider_video(video_total_frames)
    window_location = window.CurrentLocation()
    window_size = window.Size
    new_window = create_window(
        OTC_ICON,
        create_layout(graph_video, slider_video, traj_folders, traj_files),
        window_location,
        window_size,
    )
    window.close()

    return (
        video,
        video_fps,
        video_total_frames,
        a_id,
        play_video,
        ret,
        frame,
        new_window,
        graph_video,
        slider_video,
    )


def update_graph_video(graph_video, frame):
    # update_graph_video(frame)
    print("9a:" + str(now_msec()))
    imgbytes = cv2.imencode(".png", frame)[1].tobytes()
    print("9b:" + str(now_msec()))
    graph_video.delete_figure("all")  # delete previous image
    print("9c:" + str(now_msec()))
    graph_video.draw_image(data=imgbytes, location=(0, 0))  # draw new image
    print("9d:" + str(now_msec()))
    # graph_video.TKCanvas.tag_lower(a_id)  # move image to bottom
    return imgbytes


def main(sg_theme=OTC_THEME):
    # Lists
    # traj_paths = []
    traj_folders = []
    traj_files = []

    # Get initial layout and create initial window
    initial_graph_video = create_graph_video()
    initial_slider_video = create_slider_video()
    window = create_window(
        OTC_ICON,
        create_layout(
            initial_graph_video, initial_slider_video, traj_folders, traj_files
        ),
    )

    # Initialize own custom video settings
    video_loaded = False
    play_video = False
    ret = None
    frame = None
    video = None
    video_fps = None
    graph_video = None
    timestamp_last_frame = 0
    playback_speed_factor = 1
    set_frame_manually = False
    if os.path.isfile(LAST_VIDEO_PATH):
        use_initial_videopath = True
    else:
        use_initial_videopath = False

    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        print("1:" + str(now_msec()))

        # Gui stuff
        event, values = window.read(timeout=0)
        # following line removed due to error, see https://stackoverflow.com/a/49862936
        # window["-column-"].expand(True, True)
        # screen_width, screen_height = window.get_screen_size()
        if (
            event == sg.WIN_CLOSED
            or event == "Cancel"
            or event == "-button_back_to_otvision-"
        ):  # if user closes window or clicks cancel
            break
        elif event == "-button_browse_traj-":
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

        # Video stuff
        print("2:" + str(now_msec()))
        if event == "-input_video-" or use_initial_videopath:  # If user loads new video
            if values["-input_video-"] != "":
                (
                    video,
                    video_fps,
                    video_total_frames,
                    a_id,
                    play_video,
                    ret,
                    frame,
                    window,
                    graph_video,
                    slider_video,
                ) = prepare_video(
                    values["-input_video-"], window, traj_folders, traj_files
                )
                update_graph_video(graph_video, frame)
                video_loaded = True
                # Save video path to user settings otvision_user_settings file
                window["-input_video-"].update(values["-input_video-"])
                otvision_user_settings["PATHS"]["LAST_VIDEO_PATH"] = values[
                    "-input_video-"
                ]
                config.write_user_settings(otvision_user_settings)
                use_initial_videopath = False
        if video_loaded:

            print("3:" + str(now_msec()))
            # Get current frame and slider position
            frame_video = video.get(cv2.CAP_PROP_POS_FRAMES)
            frame_slider = values["-slider_video-"] + 1

            print("4:" + str(now_msec()))
            # Get state of video player
            if event == "-button_play-":
                play_video = True
            elif event == "-button_pause-" or not ret:
                play_video = False
                timestamp_current_frame = 0
            if frame_video != frame_slider:
                set_frame_manually = True
            playback_speed_factor = float(values["-spin_speed-"])

            # If a new frame has to be displayed
            print("5:" + str(now_msec()))
            if play_video or set_frame_manually:
                # log_videoplayer("Pos. 1", video, values)
                # Calculate next frame
                if set_frame_manually:
                    frame_video = frame_slider  # int(values["-slider_video-"]) + 1
                    set_frame_manually = False
                elif play_video:
                    delta_frames = video_fps / PLAYER_FPS
                    if playback_speed_factor < 0 or playback_speed_factor > 1:
                        delta_frames = delta_frames * playback_speed_factor
                    frame_video = frame_slider = frame_video + delta_frames
                print("5a:" + str(now_msec()))
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_video - 1)
                # log_videoplayer("Pos. 2", video, values)
                print("5b:" + str(now_msec()))
                window["-slider_video-"].update(frame_slider - 1)
                # log_videoplayer("Pos. 3", video, values)

                print("6:" + str(now_msec()))
                # Retrieve current frame from video using opencv cap.read()
                ret, frame = video.read()
                if ret:
                    print("7:" + str(now_msec()))
                    # Define waiting time to show frame for displaying in correct speed
                    if play_video:
                        if playback_speed_factor > 0 and playback_speed_factor < 1:
                            time_between_frames = 1 / (
                                PLAYER_FPS * playback_speed_factor
                            )
                        else:
                            time_between_frames = 1 / PLAYER_FPS
                        timestamp_current_frame = (
                            timestamp_last_frame + time_between_frames
                        )
                        if (
                            timestamp_current_frame > 0
                            and now_msec() < timestamp_current_frame
                        ):
                            pause.until(timestamp_current_frame)
                        print("8:" + str(now_msec()))
                    # log_videoplayer("Pos. 4", video, values)
                    # Get timestamp of displaying current frame
                    timestamp_last_frame = now_msec()
                    print("9:" + str(now_msec()))
                    # Display new frame
                    update_graph_video(graph_video, frame)
                    print("10:" + str(now_msec()))

    print("Huhu")
    window.close()


# To Dos:
# - Rearange loop to enable Play, Pause and Slider simultaneously
# - Load video automatically by file name of "[...]_trackspx.json"?
#   or otherwise by browse button
# - Show current video (or real?) time and total video time on slider
# - Remove error when video runs out of frames
# - Find and remove error of no more frames displaying after certain time (out of RAM?)
# - Draw on every mouse click (everey now and then it doesnt work)
# - Draw crosshair instead of circle
# - Add embededding or popup for map
# - Append clicked pixel and utm refpts to dictionary
# - Button to load/save this dictionary from/to json/pkl/npy/txt
# - Button to call transfrom_to_world function passing refpts
# - Add function for accuracy evaluation to transform_to_world
# - Some element to show results of accuracy evaluation
