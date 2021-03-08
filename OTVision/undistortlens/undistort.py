# %%
import cv2 as cv
import glob
import json
import numpy as np
import pandas as pd

# %%
# H:\06_OTCamera\OTCamera\OTCamera\data.txt

path = "H:\\06_OTCamera\\OTVision\\OTVision\\OTVision\\"


path2 = "H:\\06_OTCamera\\OTVision\\OTVision\\OTVision\\undistortlens\\data.txt"


# text_files = glob.glob(path + "/**/data.txt", recursive = True)


# cameraparams = text_files[0]


# loads dictionary with parameters from
def load_params():
    """Loads data.txt - file with the distinct camera parameters K and D

    K is the cameramatrix
    D are distance coefficent
    funtion returns dictionary
    """
    text_files = glob.glob(path2, recursive=True)

    cameraparams = text_files[0]

    with open(cameraparams) as fh:

        data = fh.read()

        params_dict = json.loads(data)

    return params_dict


# function to undistort pictures
def undistort_picture():
    """ takes params and undistorts a list of images"""

    params_dict = load_params()

    # load params from dictionary
    mtx = params_dict["K"]
    dist = params_dict["D"]

    # turn coefficents into matrix and array
    matrix_list = [mtx[0], mtx[1], mtx[2]]

    mtx = np.array(matrix_list)
    dist = np.array(dist)

    # print(mtx, dist)

    image_list = glob.glob(path + "imagefolder\\*.jpg", recursive=True)

    i = 0

    # undistort list of images with camera coefficents
    try:
        for img in image_list:

            img = cv.imread(img)

            h, w = img.shape[:2]

            newcameramtx, roi = cv.getOptimalNewCameraMatrix(
                mtx, dist, (w, h), 1, (w, h)
            )
            # undistort

            dst = cv.undistort(img, mtx, dist, None, newcameramtx)

            # crop the image
            x, y, w, h = roi
            dst = dst[y : y + h, x : x + w]

            # saves undistorted pictures to path

            cv.imwrite(
                "H:\\06_OTCamera\\OTVision\\OTVision\\OTVision\\imagefolder_undistorted\\preview{0}_undistorted.jpg".format(
                    str(i)
                ),
                dst,
            )

            i += 1
    except:

        print("no pictures to undistort")


def load_trajectories():

    # loads dataframe with inconsistent column count and trajectoriepoints
    x_res = 800
    y_res = 600

    with open(
        "H:\\06_OTCamera\\OTVision\\OTVision\\tests\\data\\testvideo_2.ottrk", "r"
    ) as json_file:
        detections = json.load(json_file)

    # extract ditcionary with frames from jsonfile
    frame_dict = detections["data"][0]

    for key in frame_dict:
        print(key, "->", frame_dict[key]["classified"])


def undistort_trajectories():
    # takes existing trajectories from dataframe und undistorts those points

    params_dict = load_params()

    # load params from dictionary
    mtx = params_dict["K"]
    dist = params_dict["D"]

    # turn coefficents into matrix and array
    matrix_list = [mtx[0], mtx[1], mtx[2]]

    mtx = np.array(matrix_list)
    dist = np.array(dist)

    # gets array of undistorted xy coordinates

    # test = np.zeros((10,1,2), dtype=np.float32)

    # xy_undistorted = cv.undistortPoints(test, mtx, dist)


if __name__ == "__main__":
    # undistort_picture()

    load_trajectories()
