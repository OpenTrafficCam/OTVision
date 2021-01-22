import cv2
import PySimpleGUI as sg
import decord


def get_psg_frame():
    window = sg.Window(
        "Demo Application - OpenCV Integration",
        [[sg.Image(filename="", key="image")]],
        location=(800, 400),
        resizable=True,
    )
    cap = cv2.VideoCapture(
        r"C:\Users\Baerwolff\Desktop\Lenovo_Arbeit\2020-02-20_Validierungsmessung_Radeberg\Videos\raspberrypi_FR20_2020-02-20_12-00-00.avi"
    )  # Setup the camera as a capture device
    return window, cap


def psg_event_loop(window, cap):
    while True:  # The PSG "Event Loop"
        event, values = window.read(
            timeout=20, timeout_key="timeout"
        )  # get events for the window with 20ms max wait
        if event is None:
            break  # if user closed window, quit
        window["image"].update(
            data=cv2.imencode(".png", cap.read()[1])[1].tobytes()
        )  # Update image in window


if __name__ == "__main__":
    window, cap = get_psg_frame()
    psg_event_loop(window, cap)
