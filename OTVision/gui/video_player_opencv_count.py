"""
Offene Aufgaben:
- EXCEL SPEICHERN KLAPPT NOCH NICHT!
- warnfenster zur erinnrung an die eingabe des modus (mit den shortcuts)
- Timestamps per Enter abfassen (in Sekunden - also mal 1.000) und an den DataFrame
übergeben
- Farbe des jeweils nächsten Zählquerschnitts ändern (und wieder zurückändern danach)
- Nach letztem Timestamp zum ersten Timestamp zurückspringen
- Videocontrols programmieren
- Gui verknüpfen
- Testen und ggf. Verbessern
- Exe erstellen
"""

import cv2
import math
import keyboard
import pandas as pd
import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog

# from tkinter import messagebox


def defineShortcuts():
    # Define shortcuts either customizable or predefined
    # print('Press and release your desired shortcut: ')
    # shortcut = keyboard.read_hotkey()
    keyboard.add_hotkey("s", saveToExcel)  # ctrl+s
    # timestampsCurrRoadUser = keyboard.add_hotkey("enter", addTimestamp)
    # keyboard.add_hotkey("p", addRoadUser, args=("Pkw", timestampsCurrRoadUser))
    # keyboard.add_hotkey("m", addRoadUser, args=("Krad", timestampsCurrRoadUser))
    # keyboard.add_hotkey("l", addRoadUser, args=("Lkw", timestampsCurrRoadUser))
    # keyboard.add_hotkey("b", addRoadUser, args=("Bus", timestampsCurrRoadUser))
    # keyboard.add_hotkey("r", addRoadUser, args=("Rad", timestampsCurrRoadUser))
    # keyboard.add_hotkey("f", addRoadUser, args=("Fuß", timestampsCurrRoadUser))


def chooseVideo():
    pass


def getVidProperties():
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


def setVidProperties():
    # Set/Change video file properties (does not always work, e.g. often only few
    # resolutions are supported)
    ret = cap.set(cv2.CAP_PROP_POS_MSEC, 2500)
    print(ret)
    ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    print(ret)
    ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
    print(ret)


def create_df_column():
    pass


def save_counters():
    pass


def clickCounter(action, x, y, flags, userdata):
    if action == cv2.EVENT_LBUTTONUP:
        print("clickCounter")
        for index, row in lines.iterrows():
            circleCenterX = int(0.5 * row["p1x"] + 0.5 * row["p2x"])
            circleCenterY = int(0.5 * row["p1y"] + 0.5 * row["p2y"])
            circleRadius = int(
                0.5
                * math.sqrt(
                    (row["p2x"] - row["p1x"]) ** 2 + (row["p2y"] - row["p1y"]) ** 2
                )
            )
            distanceCircleCenter = math.sqrt(
                (x - circleCenterX) ** 2 + (y - circleCenterY) ** 2
            )
            if distanceCircleCenter <= circleRadius:
                countingLine = row["name"]
                pressedKey2 = cv2.waitKey(0) & 0xFF
                if pressedKey2 == 112:
                    modeOfTransport = "LV"
                elif pressedKey2 == 108:
                    modeOfTransport = "SV"
                elif pressedKey2 == 114:
                    modeOfTransport = "Rad"
                elif pressedKey2 == 102:
                    modeOfTransport = "Fuss"
                elif pressedKey2 == 98:
                    modeOfTransport = "Bus"
                videoTime = cap.get(cv2.CAP_PROP_POS_MSEC)
                addRoadUser(modeOfTransport, videoTime, countingLine)


def drawLines(action, x, y, flags, userdata):
    global drawing, lineCounter, lines
    # Action to be taken when left mouse button is pressed
    if action == cv2.EVENT_LBUTTONDOWN:
        print("down")
        lineCounter += 1
        lines = lines.append(
            pd.Series([0, 0, 0, 0, 0, 0], index=lines.columns), ignore_index=True
        )
        print(lines)
        # lines.append(newLine, ignore_index=True)
        drawing = True
        lines.iloc[lineCounter - 1, 2] = x
        lines.iloc[lineCounter - 1, 3] = y
    elif action == cv2.EVENT_LBUTTONUP:
        print("up")
        drawing = False
        lines.iloc[lineCounter - 1, 4] = x
        lines.iloc[lineCounter - 1, 5] = y
        showLines(previousFrame)
        lines.iloc[lineCounter - 1, 0] = lineCounter
        # root = tk.Tk()
        # root.withdraw
        lines.iloc[lineCounter - 1, 1] = simpledialog.askstring(
            "Zählquerschnitt", "Wie ist die Bezeichnung des Zählquerschnitts?"
        )
        # root.destroy()
    elif drawing:
        print("move")
        if action == cv2.EVENT_MOUSEMOVE:
            print(lineCounter)
            lines.iloc[lineCounter - 1, 4] = x
            lines.iloc[lineCounter - 1, 5] = y
            showLines(previousFrame)


def deleteLines():
    global lines
    lines.drop(lines.index[len(lines) - 1])
    showLines(previousFrame)


def showLines(whateverFrame):
    global frame, previousFrame
    whateverFrameCopy = whateverFrame.copy()
    i = -1
    for index, row in lines.iterrows():
        i += 1
        cv2.line(
            whateverFrameCopy,
            (row["p1x"], row["p1y"]),
            (row["p2x"], row["p2y"]),
            (0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        circleCenter = (
            int(0.5 * row["p1x"] + 0.5 * row["p2x"]),
            int(0.5 * row["p1y"] + 0.5 * row["p2y"]),
        )
        circleRadius = int(
            0.5
            * math.sqrt((row["p2x"] - row["p1x"]) ** 2 + (row["p2y"] - row["p1y"]) ** 2)
        )
        cv2.circle(
            whateverFrameCopy, circleCenter, circleRadius, (0, 255, 0), thickness=2
        )
        # cv2.putText(whateverFrameCopy, str(i), (row[2], row[3]),
        # cv2.FONT_HERSHEY_COMPLEX, 1.5, (250, 10, 10), 2,
        #            cv2.LINE_AA)
    if previousFrame is not None:
        cv2.imshow("Video", whateverFrameCopy)
    # frame = whateverFrame
    # previousFrame = whateverFrame


def addTimestamp():
    global timestampCounter
    if timestampCounter == 0:
        timestampsCurrRoadUser = []
    timestampCounter += 1
    vidPosition = cap.get(cv2.CAP_PROP_POS_MSEC)
    timestampsCurrRoadUser = timestampsCurrRoadUser + [vidPosition]
    return timestampsCurrRoadUser


def addRoadUser(modeOfTransport, videoTime, countingLine):
    global timestamps, timestampCounter
    timestamps = timestamps.append(
        pd.Series([0] * 5, index=timestamps.columns), ignore_index=True
    )
    # lines = lines.append(pd.Series([0, 0, 0, 0, 0, 0], index=lines.columns),
    # ignore_index=True)
    roadUserCounter = len(timestamps.index)
    timestamps.iloc[roadUserCounter - 1, 0] = roadUserCounter
    timestamps.iloc[roadUserCounter - 1, 1] = videoFile1
    timestamps.iloc[roadUserCounter - 1, 2] = countingLine
    timestamps.iloc[roadUserCounter - 1, 3] = modeOfTransport
    timestamps.iloc[roadUserCounter - 1, 4] = videoTime
    print(timestamps)


def saveToExcel():
    # root = tk.Tk()
    # root.withdraw()
    save_name = filedialog.asksaveasfilename(
        initialdir="/", title="Select file"
    )  # filetypes=(("Excel files", "*.xlsx"), ("all files", "*.*"))'
    # root.destroy()
    print(save_name)
    timestamps.to_excel(save_name[::-5] + "_Zeitstempel.xlsx")
    lines.to_excel(save_name[::-5] + "_Zaehlquerschnitte.xlsx")


"""def askLoadOrNew():
    root = tk.Tk()
    root.withdraw()
    answerNew = messagebox.askyesno(
        "easycount timestamps",
        "Wollen Sie eine Auswertung anfangen? [Ja = neu; Nein = fortsetzen]",
    )
    root.destroy()
    if answerNew:
        askForIntersection()
    elif answerNew:
        pass
    elif answerNew is None:
        askLoadOrNew()"""


def createDataframes():
    global timestamps, lines
    # dataframe to store the timestamps
    timestamps = pd.DataFrame(
        columns=["Nr", "Videodatei1", "Zählquerschnitt", "Modus", "Zeit"]
    )

    # Lists to store the lines
    lines = pd.DataFrame(columns=["id", "name", "p1x", "p1y", "p2x", "p2y"])


def changePlaybackSpeed(action, x, y, flags, userdata):
    global timeBetwFrames, speed
    vidFPS = cap.get(cv2.CAP_PROP_FPS)  # Frame rate
    if action == cv2.EVENT_MOUSEWHEEL:
        if flags > 0 and speed < 2000:
            print(flags)
            speed += 50
            print(speed)
        elif flags < 0 and speed > 50:
            print(flags)
            speed -= 50
            print(speed)
    elif action == cv2.EVENT_LBUTTONDBLCLK:
        print("doubleklick")
    cv2.setTrackbarPos("speed", "Video", speed)
    newFPS = int(vidFPS / 100 * speed)
    if newFPS >= 0:
        timeBetwFrames = int(1000 / newFPS)


def changeVidPosition(pos):
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    # showLines(frame)


def propertiesWindow():
    """Create additional window"""
    root = tk.Tk()
    root.title("Properties")
    # label = tk.Label(root, fg="dark green")
    # label.pack()
    # counter_label(label)
    button = tk.Button(
        root, text="Create new Counter", width=25, command=root.destroy
    )  # command=root.destroy
    button.pack()
    root.mainloop()


# askLoadOrNew()
# defineShortcuts()
createDataframes()
# Define video capturing source(s)
root = tk.Tk()
root.withdraw()
videoFile1 = filedialog.askopenfilename(
    initialdir="/",
    title="Video auswählen",
    filetypes=(("Video files", "*mp4"), ("all files", "*.*")),
)

# root.destroy()
# videoFile1 = r"C:\Users\Martin\Downloads\Pexels Videos 2431853.mp4"
# cap =
# cv2.VideoCapture(r"C:\Users\Martin\Desktop\Code\OpenCV_testdata\week2\chaplin.mp4")
# Play Videofile
cap = cv2.VideoCapture(videoFile1)
# cap = cv2.VideoCapture(0) # Live Stream of Webcam (if multiple, try 1, 2, 3 etc...)
# Webcam: using waitKey(1) is good because the display frame rate will be limited by
# the frame rate of the webcam

# Check if camera opened successfully
if cap.isOpened():
    print("Error opening video stream or file")

timeBetwFrames = getVidProperties()

cv2.namedWindow("Video")

# Create trackbars
numberOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# cv2.createTrackbar('position', 'Video', 0, numberOfFrames, changeVidPosition)
# cv2.createTrackbar('speed', 'Video', 100, 1000, changePlaybackSpeed)

# Show video with a certain framerate
speed = 100
lineCounter = 0
timestampCounter = 0
roadUserCounter = 0
drawing = False
previousFrame = None
pressedKey = 0
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Exit the loop if key "esc" is hit
        cv2.setMouseCallback("Video", changePlaybackSpeed)
        # cv2.getTrackbarPos('position', 'Video')
        # cv2.getTrackbarPos('speed', 'Video')
        if pressedKey == 115:
            saveToExcel()
        if pressedKey == 27:
            break
        # Pause loop if key "space" is hit
        if pressedKey == 32:
            # cv2.getTrackbarPos('position', 'Video')
            # cv2.getTrackbarPos('speed', 'Video')
            cv2.setMouseCallback("Video", clickCounter)
            pressedKey = cv2.waitKey(0) & 0xFF
            if pressedKey == 115:
                saveToExcel()
            if pressedKey == 27:
                break
            if pressedKey == 113:  # Buchstabe
                cv2.setMouseCallback("Video", drawLines)
                print("Hello Mouse Callback")
            if pressedKey == 119:
                print("Goodbye line")
                deleteLines()
            while pressedKey != (32):
                pressedKey = cv2.waitKey(0) & 0xFF
            # Lambda, so that callback function can be used outside while loop
            cv2.setMouseCallback("Video", lambda *args: None)
        showLines(frame)
        # Wait specified milliseconds and check key hits
        pressedKey = cv2.waitKey(timeBetwFrames) & 0xFF
        # Display the resulting frame
        previousFrame = frame
        # cv2.imshow('Video', frame)
    else:
        break

# In the end always release the video capture
cap.release()
cv2.destroyAllWindows()
