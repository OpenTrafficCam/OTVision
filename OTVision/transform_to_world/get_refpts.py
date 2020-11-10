# -*- coding: utf-8 -*-
"""
Pixel coordinates of reference points are clicked in the frame (annotations
are saved as image)
Corresponding UTM world coordinates of the reference points are entered
(from maps service or own measurement)
    --> Use the following page:
    https://www.koordinaten-umrechner.de/decimal/51.000000,10.000000?karte=EsriSat&zoom=8
For each both a text file and a numpy file are saved in the same directory as
the frame

Instructions:
    Click left mouse button     Mark current mouse position
    (Space bar                   Skip point)
    Z                           Delete last marker
    S                           Save markers in new image file and text file
    Esc                         Finish input, close window
"""

import cv2
from tkinter import filedialog
from tkinter import simpledialog
import tkinter as tk
import numpy as np


# import pandas as pd


def show_markers_in_img():
    """
    Draws markers to the pixel coordinates stored in the array refptsPixel
    into the loaded image
    """
    global refptsPixel, zaehler, img, x2, y2
    img = cv2.imread(dateiname, 1)
    cv2.putText(
        img,
        "Bisher markierte Punkte:" + str(zaehler),
        (groesse_beschr, groesse_beschr * 2),
        schriftart,
        cv2.getFontScaleFromHeight(schriftart, groesse_beschr),
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    zaehler_skip = 0
    for refpt in range(0, zaehler):
        beschr = "P" + str(refpt + 1)
        x2 = int(refptsPixel[refpt, 1])
        y2 = int(refptsPixel[refpt, 2])
        px2u = (x2, y2 + groesse_marker)
        px2o = (x2, y2 - groesse_marker)
        px2l = (x2 - groesse_marker, y2)
        px2r = (x2 + groesse_marker, y2)
        # cv2.circle(img, px2, 5, (0, 0, 255), -1)
        cv2.line(img, px2u, px2o, (0, 0, 255), 1)
        cv2.line(img, px2l, px2r, (0, 0, 255), 1)
        cv2.putText(
            img,
            beschr,
            px2o,
            schriftart,
            cv2.getFontScaleFromHeight(schriftart, groesse_beschr),
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )  # siehe cv2.putText & cv2.getFontFromHeight
    cv2.imshow("image", img)


def update_array_refptsPixel(x, y, status):
    """
    todo: #9 Add function description
    """
    global refptsPixel, refptsPixel_bem, zaehler, x2, y2
    # beschr = "P"+str(zaehler + 1)
    if zaehler == 0:
        refptsPixel = np.array([[zaehler + 1, x, y, status]], dtype="float64")
        # custom = np.dtype([("refpt-ID","S20"),("Px_x","float64"),("Px_y","float64"),("Status","S20")])
        # refptsPixel_2 = np.array([beschr,x,y,status], dtype=custom)
    else:
        refptsPixel = np.append(refptsPixel, [[zaehler + 1, x, y, status]], axis=0)
        # refptsPixel_2 = np.append(refptsPixel_2,[beschr,x,y,status], axis=0)
    zaehler += 1


def update_array_refptsWorld(x_world, y_world, status):
    """
    description follows
    """
    global refptsWorld, refptsPixel_bem_world, zaehler_refptsWorld, x2_world, y2_world
    # beschr = "P"+str(zaehler + 1)
    if zaehler_refptsWorld == 0:
        refptsWorld = np.array(
            [[zaehler_refptsWorld + 1, x_world, y_world, status]], dtype="float64"
        )
        # custom = np.dtype([("refpt-ID","S20"),("Px_x","float64"),("Px_y","float64"),("Status","S20")])
        # refptsPixel_2 = np.array([beschr,x,y,status], dtype=custom)
    else:
        refptsWorld = np.append(
            refptsWorld, [[zaehler_refptsWorld + 1, x_world, y_world, status]], axis=0
        )
        # refptsPixel_2 = np.append(refptsPixel_2,[beschr,x,y,status], axis=0)
    zaehler_refptsWorld += 1


def inputbox_longitude():
    """Get longitude or east value in UTM world coordinates from user input"""
    longitude = simpledialog.askfloat(
        "Longitude",
        "Please enter the UTM longitude (X or east value) of the of the previously "
        "clicked reference point as point separated decimal (6 pre-decimal places)",
        minvalue=100000,
        maxvalue=10000000,
    )
    if longitude is not None:
        print("The longitude is ", longitude)
    else:
        print("You didn't enter a longitude")
        return inputbox_longitude()
    return longitude


def inputbox_latitude():
    """Get latitude or north value in UTM world coordinates from user input"""
    latitude = simpledialog.askfloat(
        "Latitude",
        "Please enter the UTM latitude (Y or north value) of the of the previously "
        "clicked reference point as point separated decimal (7 pre-decimal places)",
        minvalue=100000,
        maxvalue=10000000,
    )
    if latitude is not None:
        print("The latitude is ", latitude)
    else:
        print("You didn't enter a latitude")
        return inputbox_latitude()
    return latitude


def mouse_drawing(event, x, y, flags, params):
    """
    Reads the current mouse position with a left click and writes it to the
    end of the array refptsPixel and adds 1 to the counter
    """
    global refptsPixel, refptsPixel_2, zaehler, x2, y2
    if event == cv2.EVENT_LBUTTONDOWN:
        status = 1
        update_array_refptsPixel(x, y, status)
        x_world = inputbox_longitude()
        y_world = inputbox_latitude()
        update_array_refptsWorld(x_world, y_world, status)
        print("Left click")


def skip_marker():
    """Fügt bei Drücken der Leertaste ein Element mit den Werten [0,0] in das Array refptsPixel ein"""
    global refptsPixel, refptsPixel_bem, zaehler, x2, y2
    status = 0
    update_array_refptsPixel(0, 0, status)
    print("SpacePress")


def delete_last_marker():
    """Löscht den letzten Merker aus dem Array"""
    global refptsPixel, refptsWorld, zaehler, zaehler_refptsWorld, x2, y2
    if zaehler == 0:
        pass
    # elif zaehler == 1:
    # refptsPixel = np.array([[0,0]], dtype="float64")
    else:
        refptsPixel = np.delete(refptsPixel, zaehler - 1, axis=0)
        refptsWorld = np.delete(refptsPixel, zaehler_refptsWorld - 1, axis=0)
        # refptsPixel_bem = np.delete(refptsPixel_bem,zaehler-1, axis=0)
        zaehler -= 1
    print("ZPress")


def save():
    """
    Array in Textdatei und Bild mit markierten Punkten in gleichen Ordner wie das 
    Ursprungsbild speichern
    """
    # refptsPixel_id = np.arange(1,zaehler+1)
    cv2.imwrite(dateiname[:-4] + "_refptsPixel.png", img)
    np.savetxt(
        dateiname[:-4] + "_refptsPixel.txt",
        refptsPixel[::, 1:3:],
        fmt="%1d",
        delimiter=";",
    )
    np.save(dateiname[:-4] + "_refptsPixel", refptsPixel[::, 1:3:])
    np.savetxt(
        dateiname[:-4] + "_refptsWorld.txt",
        refptsWorld[::, 1:3:],
        fmt="%1.12f",
        delimiter=";",
    )
    np.save(dateiname[:-4] + "_refptsWorld", refptsWorld[::, 1:3:])

    # print(refptsPixel)
    # print(refptsPixel_2)
    pass


root = tk.Tk()
root.withdraw()
root.filename = filedialog.askopenfilename(
    initialdir="/",
    title="Select file",
    filetypes=(("png files", "*.png"), ("jpeg files", "*.jpg"), ("all files", "*.*")),
)
dateiname = root.filename  # "Sleepers.png"
img = cv2.imread(dateiname, 1)
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_drawing)

zaehler = 0
zaehler_refptsWorld = 0

schriftart = cv2.FONT_ITALIC
groesse_beschr_rel = 0.02
groesse_marker_rel = 0.02
groesse_beschr = round(img.shape[0] * groesse_beschr_rel)
groesse_marker = round(img.shape[0] * groesse_marker_rel)

while True:
    # print(zaehler)
    show_markers_in_img()
    key = cv2.waitKey(
        1
    )  # bei waitKey(0) würden beim Klicken keine roten Kreuze eigezeichnet werden
    if key == 27:  # Fenster schließen
        break
    elif key == 32:  # Markierung überspringen
        pass  # Skip_marker()
    elif key == 122 or key == 90:  # letzte Markierung löschen
        delete_last_marker()
    elif key == 115 or key == 83:  # In Textdatei speichern und schließen
        save()
        break
    elif key == ord("d"):
        refptsPixel = np.empty((1, 2), dtype="float64")

cv2.destroyAllWindows()
root.destroy()
print(refptsPixel)
refptsPixel = np.empty((1, 2), dtype="float64")
refptsWorld = np.empty((1, 2), dtype="float64")

"""
Aufgaben:
    Nach Dateipfad browsen
    Ergebnisse dann in gleichen Ordner speichern
"""
