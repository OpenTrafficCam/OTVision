import cv2
import math
import keyboard
import pandas as pd


MOUSEWHEEL_MODE = True
MAGNIFIER_MODE = False
DRAWING_MODE = "points"


def getVidProperties(cap):
    # Read video file properties like current position, resolution, framerate, codec
    vidPosition = cap.get(
        cv2.CAP_PROP_POS_MSEC
    )  # Current position of the video file in milliseconds
    vidWidth = cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )  # Width of the frames in the video stream
    vidHeight = cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )  # Height of the frames in the video stream
    vidFPS = cap.get(cv2.CAP_PROP_FPS)  # Frame rate
    vidCodec = cap.get(cv2.CAP_PROP_FOURCC)  # 4-character code of codec

    print("Position = " + str(vidPosition))
    print("Breite = " + str(vidWidth))
    print("Höhe = " + str(vidHeight))
    print("Framerate = " + str(vidFPS))
    print(vidCodec)

    # Calculate how long to wait between frames to playback in correct video framerate
    # (own addition)
    timeBetwFrames = int(1000 / vidFPS)
    print(timeBetwFrames)
    return timeBetwFrames


def on_trackbar(val):
    # Corrupts video feed
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    print(val)


def control_callback(action, x, y, flags, userdata):
    print("Control")
    global cap
    if MOUSEWHEEL_MODE and action == cv2.EVENT_MOUSEWHEEL:
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if flags > 2000000:
            print(flags)
            current_frame += 100
        elif flags < -2000000:
            print(flags)
            current_frame -= 100
        if flags > 0:
            print(flags)
            current_frame += 5
        if flags < 0:
            print(flags)
            current_frame -= 5
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)


def drawing_callback(action, x, y, flags, userdata):
    control_callback(action, x, y, flags, userdata)
    global cap, point_1, lines, i
    if DRAWING_MODE == "points":
        if action == cv2.EVENT_LBUTTONUP:
            points.loc[len(points.index)] = [x, y]  # Append point to dataframe
        elif action == cv2.EVENT_RBUTTONUP:
            points.drop(points.tail(1).index, inplace=True)  # Delete last point from df
        print(points)
        show_elements_on_video_player()
    if DRAWING_MODE == "lines":
        if action == cv2.EVENT_LBUTTONDOWN:
            point_1 = [x, y]
            i = len(lines.index)
        elif action == cv2.EVENT_MOUSEMOVE and point_1 is not None:
            print("Mousemove")
            line = point_1 + [x, y]
            lines.loc[i] = line  # Append line to dataframe
        elif action == cv2.EVENT_LBUTTONUP:
            line = point_1 + [x, y]
            point_1 = None
            lines.loc[i] = line  # Append line to dataframe
        elif action == cv2.EVENT_RBUTTONUP:
            lines.drop(lines.tail(1).index, inplace=True)  # Delete last line from df
        print(lines)
        show_elements_on_video_player()
    if action == cv2.EVENT_MOUSEMOVE:
        # show_elements_on_video_player()
        pass


def show_elements_on_video_player():
    frame_copy = frame.copy()
    if DRAWING_MODE == "points":
        for index, row in points.iterrows():
            cv2.circle(
                img=frame_copy,
                center=(row["px_x"], row["px_y"]),
                radius=5,
                color=(0, 255, 0),
                thickness=2,
            )
            # Color is not yet working
            """cv2.putText(
                img=frame_copy,
                text=index + 1,
                org=(row["px_x"], row["px_y"]),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                color=(255, 255, 0),
            )"""
    if DRAWING_MODE == "lines":
        for index, row in lines.iterrows():
            cv2.line(
                img=frame_copy,
                pt1=(row["px_x1"], row["px_y1"]),
                pt2=(row["px_x2"], row["px_y2"]),
                color=(0, 255, 0),
                thickness=2,
            )
    cv2.imshow(video_player_window_title, frame_copy)


def main():
    global cap, video_player_window_title, frame, test, points, lines
    TEST_VIDEO = r"C:\Users\Baerwolff\Desktop\Lenovo_Arbeit\2020-02-20_Validierungsmessung_Radeberg\Videos\raspberrypi_FR20_2020-02-20_12-00-00.mkv"
    cap = cv2.VideoCapture(TEST_VIDEO)
    video_player_window_title = "Video Player"
    test = cv2.namedWindow(video_player_window_title)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    """cv2.createTrackbar(
        "Frames", video_player_window_title, 0, total_frames, on_trackbar
    )"""
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    time_between_frames = int(1000 / video_fps)
    pressed_key = None
    if DRAWING_MODE == "points":
        points = pd.DataFrame(columns=["px_x", "px_y"])
    if DRAWING_MODE == "lines":
        lines = pd.DataFrame(columns=["px_x1", "px_y1", "px_x2", "px_y2"])
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.setMouseCallback(video_player_window_title, control_callback)
            show_elements_on_video_player()
            # cv2.imshow(video_player_window_title, frame)
            """cv2.setTrackbarPos(
                "Frames",
                video_player_window_title,
                int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            )"""
            if pressed_key == 32:
                cv2.setMouseCallback(video_player_window_title, drawing_callback)
                print("Hello Drawing Callback")
                pressed_key = cv2.waitKey(0) & 0xFF
                while pressed_key != (32):
                    pressed_key = cv2.waitKey(0) & 0xFF
            if pressed_key == 27:
                break
        pressed_key = cv2.waitKey(time_between_frames) & 0xFF
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
