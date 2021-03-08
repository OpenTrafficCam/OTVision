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

    df = pd.DataFrame()

    # loads dataframe with inconsistent column count and trajectoriepoints

    with open(
        "H:\\06_OTCamera\\OTVision\\OTVision\\OTVision\\trajectories\\example_traj.csv",
        "r",
    ) as f:

        for line in f:
            df = pd.concat(
                [df, pd.DataFrame([tuple(line.strip().split(";"))])], ignore_index=True
            )

    df = df.iloc[1:, 9:]

    df_x = df.iloc[:, 1::5]

    df_y = df.iloc[:, 2::5]

    for i in range(10):
        np.array(df.iloc[0:1, :])

    # print(df_x, df_y)


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
    undistort_picture()

    # load_trajectories()
