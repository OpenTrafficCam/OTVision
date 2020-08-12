"""
This module calculates a homography matrix and transforms trajectory points from pixel into world coordinates.
It requires reference points in both pixel and world coordinates as well as trajectories in pixel coordinates.
Reference points have to be loaded from .npy files (or from semicolon delimited .txt files without headers or indices).
Multiple trajectory files with data arranged in otc style (OpenTrafficCam) can be selected,  preferably from .pkl files.
The module saves the trajectories in world coordinates as .pkl files.
For different camera views, the module has to be run repeatedly with respective reference points in pixel coordinates.

Additional information due to OpenCV shortcoming:
Some extra calculations are performed, because OpenCV functions cannot deal with all digits of UTM coordinates.
Hence, from "refptsWorld" only the three digits before and all digits after delimiter are used for transformation.
After transformation the truncated thousands digits are concatenated again.
To avoid errors due to different thousands digits within study area, before transformation "refptsWorld" are shifted,
so that the center of all reference points lies on a round 500 m value for both x & y UTM coordinates.
After transformation the "trajWorld" coordinates are shifted back by exactly the same value.

To Dos:
- Save also homography as npy file?
"""

import cv2
from tkinter import filedialog
import numpy as np
import pandas as pd


# test paths
traj_path_TEST = r"C:\Users\Baerwolff\Desktop\Lenovo_Arbeit\2020-02-20_Validierungsmessung_Radeberg\DataFromSky_Tracks\Test_5min\viewer_exports"
refpts_path_TEST = r"C:\Users\Baerwolff\Desktop\Lenovo_Arbeit\2020-02-20_Validierungsmessung_Radeberg\homography"


def read_refptsPixel_dialog(path_dialog_default=refpts_path_TEST):
    """User can select one file containing reference points in pixel coordinates in npy or csv format
    (generated with getRefPts.py) and they are read to a numpy array"""
    try:
        refptsPixel_path = filedialog.askopenfilename(initialdir=path_dialog_default,
                                              title="Select reference points in pixel coordinates (.txt or .npy)",
                                              filetypes=(("Numpy files", "*.npy"),
                                                         ("Text files", "*.txt"),
                                                         ("all files", "*.*")))
        if refptsPixel_path.endswith(".npy"):
            print(refptsPixel_path + " is a numpy file")
            refptsPixel = np.load(refptsPixel_path)
        elif refptsPixel_path.endswith(".txt"):
            print(refptsPixel_path + " is a text file")
            refptsPixel = np.loadtxt(refptsPixel_path, dtype="i4", delimiter=";")
    except:
        print("Please try again")
        return read_refptsPixel_dialog(path_dialog_default)
    return refptsPixel


def read_refptsWorld_dialog(path_dialog_default=refpts_path_TEST):
    """User can select one file containing reference points in world coordinates in npy or csv format
    (generated with getRefPts.py) and they are read to a numpy array"""

    try:
        refptsWorld_path = filedialog.askopenfilename(initialdir=path_dialog_default,
                                              title="Select reference points in World coordinates (.txt or .npy)",
                                              filetypes=(("Numpy files", "*.npy"),
                                                         ("Text files", "*.txt"),
                                                         ("all files", "*.*")))
        if refptsWorld_path.endswith(".npy"):
            print(refptsWorld_path + " is a numpy file")
            refptsWorld = np.load(refptsWorld_path)
        elif refptsWorld_path.endswith(".txt"):
            print(refptsWorld_path + " is a text file")
            refptsWorld = np.loadtxt(refptsWorld_path, delimiter=";")
    except:
        print("Please try again")
        return read_refptsWorld_dialog(path_dialog_default)
    return refptsWorld


def choose_trajPixel_dialog(path_dialog_default=traj_path_TEST):
    """User can select one or multiple trajectory files in pkl or csv format"""
    try:
        trajPixel_paths = filedialog.askopenfilenames(initialdir=path_dialog_default,
                                              title="Select converted DataFromSky trajectories in pixel coordinates (.npy)",
                                              filetypes=(("Python pickle files", "*.pkl"),
                                                         ("CSV files", "*.csv"),
                                                         ("All files", "*.*")))
    except:
        print("Please try again")
        return read_trajPixel_dialog(path_dialog_default)
    return trajPixel_paths

def read_trajPixel_dialog(trajPixel_path):
    """Read a single trajectory file in pkl or csv format (otc style) to a pandas dataframe"""
    try:
        if trajPixel_path.endswith(".pkl"):
            print(trajPixel_path + " is a python pickle file")
            trajPixel = pd.read_pickle(trajPixel_path)
            print(trajPixel)
        elif trajPixel_path.endswith(".csv"):
            print(trajPixel_path + " is a csv file")
            trajPixel = pd.read_csv(trajPixel_path, delimiter=";", index_col=0)
            print(trajPixel)
        return trajPixel
    except:
        pass


def calculateHomographyMatrix(refptsPixel, refptsWorld):
    """Calculatiing homography matrix using pixel and world coordinates of corresponding reference points"""

    # Upshift both y and y world coordinates of reference points to next round 500m value (UTM is in meters)
    min = np.amin(refptsWorld, axis=0)
    max = np.amax(refptsWorld, axis=0)
    mean = np.divide(np.add(min, max), 2)
    mean_vorkomma = mean.astype(int)
    mean_vorkommaPt1 = np.divide(mean_vorkomma, 1000).astype(int)
    mean_vorkommaPt1_Plus_500 = np.add(mean_vorkommaPt1.astype(float), 0.5)
    mean_Plus_500 = np.multiply(mean_vorkommaPt1_Plus_500, 1000)
    upshiftWorld = np.subtract(mean_Plus_500, mean)
    refptsWorld_upshifted = np.add(refptsWorld, upshiftWorld)

    # Truncate thousands digits from shifted reference points
    refptsWorld_upshifted_nachkomma = np.mod(refptsWorld_upshifted, 1)
    refptsWorld_upshifted_vorkomma = refptsWorld_upshifted.astype(int)
    refptsWorld_upshifted_vorkommaPt1 = np.divide(refptsWorld_upshifted_vorkomma, 1000).astype(int)
    refptsWorld_upshifted_vorkommaPt1_1row = np.array([[refptsWorld_upshifted_vorkommaPt1.item(0), refptsWorld_upshifted_vorkommaPt1.item(1)]])
    refptsWorld_upshifted_vorkommaPt2 = np.mod(refptsWorld_upshifted_vorkomma, 1000)
    refptsWorld_upshifted_zerlegt = np.add(refptsWorld_upshifted_vorkommaPt2, refptsWorld_upshifted_nachkomma)

    # Calculate homography matrix with refpts in pixel coordinates and truncated & shifted refpts in world coordinates
    homographyMatrix, mask = cv2.findHomography(refptsPixel, refptsWorld_upshifted_zerlegt, cv2.RANSAC, 3.0)  # RANSAC: Otulier/Inlier definieren??? # FEHLER:
    print(homographyMatrix)
    return homographyMatrix, refptsWorld_upshifted_vorkommaPt1_1row, upshiftWorld


def convertPixelToWorld(trajPixel, homographyMatrix, refptsWorld_upshifted_vorkommaPt1_1row, upshiftWorld):
    """Convert trajectories from pixel to world coordinates using homography matrix"""

    # Transform pandas dataframe to numpy array, add 1 dimension and apply OpenCV´s perspective transformation
    trajPixelNp = trajPixel[['x', 'y']].to_numpy(dtype='float32')
    trajPixelNpTmp = np.array([trajPixelNp], dtype='float32')
    trajWorld_upshifted_Np_zerlegt_3d = cv2.perspectiveTransform(trajPixelNpTmp, homographyMatrix)
    trajWorld_upshifted_Np_zerlegt = np.squeeze(trajWorld_upshifted_Np_zerlegt_3d)

    # Concatenate the thousands digits truncated before transformation
    trajWorld_upshifted_Np = np.add(np.multiply(refptsWorld_upshifted_vorkommaPt1_1row, 1000),
                                    trajWorld_upshifted_Np_zerlegt)

    # Shift back down both y and y world coordinates (same values as reference points were upshifted)
    trajWorldNp = np.subtract(trajWorld_upshifted_Np, upshiftWorld)

    # In trajectory dataframe, overwrite pixel coordinates with world coordinates (from numpy array)
    trajWorld = trajPixel
    trajWorld[['x', 'y']] = trajWorldNp

    return trajWorld

def saveTrajWorld(trajPixel_path, trajWorld):
    """Save trajectories in world coordinates as python pickle files and csv files"""

    trajWorld.to_csv(trajPixel_path[:-4] + "World_decDot.csv", index=False, sep=";")
    trajWorld.to_csv(trajPixel_path[:-4] + "World_decComma.csv", index=False, sep=";", decimal=",")
    trajWorld.to_pickle(trajPixel_path[:-4] + "World.pkl")


def main():
    # Find homography matrix for corresponding refpts in pixel and world coordinates
    refptsPixel = read_refptsPixel_dialog()
    refptsWorld = read_refptsWorld_dialog()
    homographyMatrix, refptsWorld_upshifted_vorkommaPt1_1row, upshiftWorld = calculateHomographyMatrix(refptsPixel, refptsWorld)
    
    # Convert trajectories from pixel to world coordinates using homography matrix for all selected trajectory files
    trajPixel_paths = choose_trajPixel_dialog()
    for trajPixel_path in trajPixel_paths:
        try:
            trajPixel = read_trajPixel_dialog(trajPixel_path)
        except:
            print("Failed to read file: " + trajPixel_path)
        trajWorld = convertPixelToWorld(trajPixel, homographyMatrix, refptsWorld_upshifted_vorkommaPt1_1row, upshiftWorld)
        saveTrajWorld(trajPixel_path, trajWorld)


if __name__ == '__main__':
    main()
