import os
import cv2
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
PLAYER_FPS = 10
otvision_user_settings = config.read_user_settings()
try:
    LAST_VIDEO_PATH = otvision_user_settings["PATHS"]["LAST_VIDEO_PATH"]
except KeyError:
    LAST_VIDEO_PATH = ""


def now_msec():
    return datetime.datetime.timestamp(datetime.datetime.now())


"""def prepare_video(path, window):
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
    graph_video = create_graph_video(video_width, video_height)
    slider_video = create_slider_video(video_total_frames)
    new_window = update_window(new_layout=layout, old_window=window)

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
    return slider_video"""


class VideoPlayer():
    def __init__(self):
        pass

    def write_frame_to_graph(self, graph_video, self.frame, shown_video_width, shown_video_height):
        # write_frame_to_graph(frame)
        print("9a:" + str(now_msec()))
        self.frame = cv2.resize(self.frame, (self.shown_video_width, self.shown_video_height))
        self.imgbytes = cv2.imencode(".png", self.frame)[1].tobytes()
        print("9b:" + str(now_msec()))
        self.graph_video.delete_figure("all")  # delete previous image
        print("9c:" + str(now_msec()))
        self.graph_video.draw_image(data=imgbytes, location=(0, 0))  # draw new image
        print("9d:" + str(now_msec()))
        # graph_video.TKCanvas.tag_lower(a_id)  # move image to bottom
        return imgbytes

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
        graph_video = sg.Graph(
            canvas_size=(400, 300),
            graph_bottom_left=(0, 400),
            graph_top_right=(300, 0),
            key="-graph_video-",
            enable_events=True,
            drag_submits=True,
            background_color="black",
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
            [browse_video],
            [input_video],
            [graph_video],
            [button_play, button_pause, text_speed, spin_speed, text_video_time],
            [slider_video],
        ]

        intialize_video_settings()

        return layout

    def intialize_video_settings():
        global video_loaded, play_video, ret, frame, video
        global timestamp_last_frame, playback_speed_factor
        # Initialize own custom video settings
        video_loaded = False
        play_video = False
        ret = None
        frame = None
        video = None
        timestamp_last_frame = 0
        playback_speed_factor = 1

    def load_video(window, values):
        global video, video_loaded, play_video, a_id
        if values["-input_video-"] != "":
            try:
                # Get video and properties with opencv
                video = cv2.VideoCapture(values["-input_video-"])
                video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
                video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
                video_ratio = video_width / video_height
                video_total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
                ret, frame = video.read()

                # Update gui elements
                window["-graph_video-"].expand(expand_x=True)
                shown_video_width = window["-graph_video-"].get_size()[0]
                shown_video_height = shown_video_width / video_ratio
                window["-graph_video-"].set_size(
                    (shown_video_width, shown_video_height)
                )  # https://github.com/PySimpleGUI/PySimpleGUI/issues/2842
                write_frame_to_graph(
                    window["-graph_video-"],
                    frame,
                    int(shown_video_width),
                    int(shown_video_height),
                )
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
            except:
                video = None
                ret = None
                print("Video from path in input field could not be loaded!")

            return ret

    def video_events(
        window, event, values, play_video=False, timestamp_last_frame=None
    ):
        print(video_loaded)
        print(video)
        if event == "-input_video-":  # If user loads new video
            ret = load_video(window, values)
        elif video_loaded and video is not None:

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
                if ret:
                    # Define waiting time to show frame for displaying in correct speed
                    if play_video:
                        if playback_speed_factor > 0 and playback_speed_factor < 1:
                            time_between_frames = 1 / (
                                PLAYER_FPS * playback_speed_factor
                            )
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
                    write_frame_to_graph(
                        window["-graph_video-"],
                        frame,
                        shown_video_width,
                        shown_video_height,
                    )

        return window, video, timestamp_last_frame


def main():
    layout = get_layout()
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
        window, video, timestamp_last_frame = video_events(window, event, values)


if __name__ == "__main__":
    main()
