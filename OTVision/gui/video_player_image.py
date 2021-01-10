import os
from tkinter import PhotoImage
from PySimpleGUI.PySimpleGUI import Image
import cv2
from PIL import Image as PIL_Image
from PIL import ImageTk as PIL_ImageTk
import PySimpleGUI as sg
import datetime
import config
import pause
from gui.update_window import update_window
from gui.sg_otc_theme import (
    OTC_ICON,
    OTC_BUTTON,
    OTC_THEME,
    OTC_FONT,
    OTC_FONTSIZE,
)


# Constants
WIDTH_COL1 = 150
sg.SetOptions(font=(OTC_FONT, OTC_FONTSIZE))
PLAYER_FPS = 2
otvision_user_settings = config.read_user_settings()
try:
    LAST_VIDEO_PATH = otvision_user_settings["PATHS"]["LAST_VIDEO_PATH"]
except KeyError:
    LAST_VIDEO_PATH = ""


def now_msec():
    global last_now
    now = datetime.datetime.timestamp(datetime.datetime.now())
    try:
        delta = now - last_now
        last_now = now
        return delta
    except:
        last_now = now
        return now


def write_frame_to_graph(graph_video, frame):
    # write_frame_to_graph(frame)
    print("Start: " + str(now_msec()))
    frame = cv2.resize(frame, (shown_video_width, shown_video_height))
    print("cv2.resize: " + str(now_msec()))

    # PIL option
    """img = PIL_Image.fromarray(frame)
    print("PIL_Image.fromarray: " + str(now_msec()))
    imgbytes = PIL_ImageTk.PhotoImage(img)
    print("PIL_ImageTk.PhotoImage: " + str(now_msec()))"""

    # cv2.imwrite option
    """cv2.imwrite("tmp.png", frame)
    print("cv2.imwrite: " + str(now_msec()))
    graph_video.draw_image(filename=r"tmp.png", location=(0, 0))  # draw new image
    print("graph_video.draw_image(filename): " + str(now_msec()))"""

    # cv2.imencode bmp and PIL ImageTk.BitmapImage (doesnt work)
    """img = cv2.imencode(".bmp", frame)[1]
    print("cv2.imencode: " + str(now_msec()))
    imgbytes = PIL_ImageTk.BitmapImage(img)
    print("PIL_ImageTk.BitmapImage: " + str(now_msec()))"""

    # cv2.imencode png or jpg
    # imgbytes = cv2.imencode(".jpeg", frame)[1].tobytes()
    img = cv2.imencode(".png", frame)[1]
    print("cv2.imencode: " + str(now_msec()))
    imgbytes = img.tobytes()
    print("img.tobytes: " + str(now_msec()))

    """graph_video.delete_figure("all")  # delete previous image
    print("graph_video.delete_figure: " + str(now_msec()))
    graph_video.draw_image(data=imgbytes, location=(0, 0))  # draw new image
    print("graph_video.draw_image(data): " + str(now_msec()))"""

    # Update image element
    graph_video.Update(data=imgbytes)
    print("graph_video.Update: " + str(now_msec()))

    # Draw frame option (doesnt work)
    """graph_video.draw_image(data=frame, location=(0, 0))  # draw new image"""
    # graph_video.TKCanvas.tag_lower(a_id)  # move image to bottom
    # return imgbytes


def get_frame():
    frame_video_player = sg.Frame("Video Player", get_layout())
    return frame_video_player


def get_layout(last_video_path=LAST_VIDEO_PATH, widt_col_1=WIDTH_COL1):
    input_video = sg.In(
        last_video_path,
        key="-input_video-",
        size=(widt_col_1, 1),
        enable_events=True,
        visible=False,  # Maybe turn to True later
    )
    browse_video = sg.FileBrowse(
        "Choose video",
        key="-browse_video-",
        target="-input_video-",
        enable_events=True,
    )
    # As sg.image doesnt support mouse callbacks I decided for sg.graph
    graph_video = sg.Image(
        size=(400, 300),
        # graph_bottom_left=(0, 400),
        # graph_top_right=(300, 0),
        key="-graph_video-",
        enable_events=True,
        # drag_submits=True,
        background_color="black",
    )
    slider_graph_video_size = sg.Slider(
        range=(0, 200),
        default_value=100,
        size=(25, 10),
        orientation="h",
        key="-slider_graph_video_size-",
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
    slider_video = sg.Slider(
        range=(0, 1), size=(60, 10), orientation="h", key="-slider_video-",
    )
    layout = [
        [browse_video, slider_graph_video_size],
        [input_video],
        [graph_video],
        [button_play, button_pause, text_speed, spin_speed, text_video_time],
        [slider_video],
    ]

    intialize_video_settings()

    return layout


def intialize_video_settings():
    global video_loaded, play_video, ret, frame, video, first_loop
    global timestamp_last_frame, playback_speed_factor
    # Initialize own custom video settings
    video_loaded = False
    play_video = False
    ret = None
    frame = None
    video = None
    timestamp_last_frame = 0
    playback_speed_factor = 1
    first_loop = True


def print_shown_video_size(shown_video_width, shown_video_height):
    print(
        "shown_video_width: "
        + str(shown_video_width)
        + ", shown_video_height: "
        + str(shown_video_height)
    )


def update_graph_size(graph_video, graph_video_zoom=100, video=None, frame=None):
    global shown_video_height, shown_video_width
    if video is not None:
        video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        video_ratio = video_width / video_height
    else:
        video_ratio = 4 / 3
    screen_width, screen_height = sg.Window.get_screen_size()
    shown_video_height = int(0.66 * graph_video_zoom / 100 * screen_height)
    shown_video_width = int(shown_video_height * video_ratio)
    # graph_video.expand(expand_x=True)
    # shown_video_width = int(graph_video.get_size()[0])
    # shown_video_height = int(graph_video.get_size()[1])
    graph_video.set_size(
        (shown_video_width, shown_video_height)
    )  # https://github.com/PySimpleGUI/PySimpleGUI/issues/2842
    if frame is not None:
        write_frame_to_graph(graph_video, frame)


def load_video(window, values):
    global video, video_loaded, play_video, a_id, last_frame
    if values["-input_video-"] != "":
        # try:
        # Get video and properties with opencv
        video = cv2.VideoCapture(values["-input_video-"])
        video_total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        ret, frame = video.read()
        last_frame = frame

        # Update gui elements
        update_graph_size(
            window["-graph_video-"], values["-slider_graph_video_size-"], video, frame,
        )
        write_frame_to_graph(window["-graph_video-"], frame)
        window["-slider_video-"].update(
            range=(0, video_total_frames)
        )  # https://github.com/PySimpleGUI/PySimpleGUI/issues/2442

        # Save video path to imput and user settings otvision_user_settings file
        window["-input_video-"].update(values["-input_video-"])
        # otvision_user_settings["PATHS"]["LAST_VIDEO_PATH"] = values["-input_video-"]
        # config.write_user_settings(otvision_user_settings)

        # Set variables
        video_loaded = True
        # use_initial_videopath = False
        a_id = None
        play_video = False
        """except:
            video = None
            ret = None
            print("Video from path in input field could not be loaded!")"""

        return ret


def video_events(window, event, values):
    global first_loop, last_frame, play_video, timestamp_last_frame
    if first_loop:
        update_graph_size(window["-graph_video-"])
        last_frame = None
        play_video = False
        first_loop = False
    if event == "-slider_graph_video_size-":
        update_graph_size(
            window["-graph_video-"],
            values["-slider_graph_video_size-"],
            video,
            last_frame,
        )
    if event == "-input_video-":  # If user loads new video
        ret = load_video(window, values)
    elif video_loaded and video is not None:
        if event == "-slider_graph_video_size-":
            update_graph_size(
                window["-graph_video-"],
                values["-slider_graph_video_size-"],
                video,
                last_frame,
            )

        # Get current frame and slider position
        frame_video = video.get(cv2.CAP_PROP_POS_FRAMES)
        frame_slider = values["-slider_video-"] + 1

        # Get state of video player
        if event == "-button_play-":
            play_video = True
        elif event == "-button_pause-" or timestamp_last_frame is None:
            play_video = False
            timestamp_current_frame = 0
        if frame_video != frame_slider:
            set_frame_manually = True
        else:
            set_frame_manually = False
        playback_speed_factor = float(values["-spin_speed-"])

        # If a new frame has to be displayed
        if play_video or set_frame_manually:
            # Calculate next frame
            if set_frame_manually:
                frame_video = frame_slider  # int(values["-slider_video-"]) + 1
                set_frame_manually = False
            elif play_video:
                video_fps = video.get(cv2.CAP_PROP_FPS)
                delta_frames = video_fps / PLAYER_FPS
                if playback_speed_factor < 0 or playback_speed_factor > 1:
                    delta_frames = delta_frames * playback_speed_factor
                frame_video = frame_slider = frame_video + delta_frames
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_video - 1)
            window["-slider_video-"].update(frame_slider - 1)

            # Retrieve current frame from video using opencv cap.read()
            ret, frame = video.read()
            last_frame = frame
            if ret:
                # Define waiting time to show frame for displaying in correct speed
                if play_video:
                    if playback_speed_factor > 0 and playback_speed_factor < 1:
                        time_between_frames = 1 / (PLAYER_FPS * playback_speed_factor)
                    else:
                        time_between_frames = 1 / PLAYER_FPS
                    if timestamp_last_frame is not None:
                        timestamp_current_frame = (
                            timestamp_last_frame + time_between_frames
                        )
                    else:
                        timestamp_current_frame = now_msec() + time_between_frames
                    if (
                        timestamp_current_frame > 0
                        and now_msec() < timestamp_current_frame
                    ):
                        pause.until(timestamp_current_frame)
                # Get timestamp of displaying current frame
                timestamp_last_frame = now_msec()
                # Display new frame
                write_frame_to_graph(window["-graph_video-"], frame)

    return window


def main():
    layout = [[get_frame()]]
    window = sg.Window(
        title="Video player",
        layout=layout,
        icon=OTC_ICON,
        location=(0, 0),
        resizable=True,
    )
    while True:
        event, values = window.read(timeout=0)
        if (
            event == sg.WIN_CLOSED
            or event == "Cancel"
            or event == "-button_back_to_otvision-"
        ):  # if user closes window or clicks cancel
            break
        window = video_events(window, event, values)


if __name__ == "__main__":
    main()
