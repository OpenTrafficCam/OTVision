import cv2 as cv
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pathlib import Path

# H:\06_OTCamera\OTCamera\OTCamera\data.txt

path = "H:\\06_OTCamera\\OTVision\\OTVision\\OTVision\\"


path2 = "H:\\06_OTCamera\\OTVision\\OTVision\\OTVision\\undistortlens\\data.txt"


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
    except KeyError:
        print("no pictures to undistort")


def get_resolution(file):
    """loads json file from tracking algorithm
    if x and y are normalized function returns height and width as integer
    if not, xy-resolution is 1
    this is necessary to calculate the absolute xy coordinates

    takes filepath of tracking data

    returns resolution as height and with or 1
    """

    with open(file, "r") as json_file:
        detections = json.load(json_file)

    # returns bool
    normalized = detections["det_config"]["normalized"]

    if normalized:
        x_res = int(detections["vid_config"]["height"])
        y_res = int(detections["vid_config"]["width"])
    else:
        x_res = 1
        y_res = 1

    return x_res, y_res


def load_trajectories(file):
    """loads json file from tracking algorithm
    converts relatives coordinates to absolut coordinates
    in px by using a given resolution
    calculates middle of bounding box

    returns array with points to undistort
    """
    x_res, y_res = get_resolution(file)

    with open(file, "r") as json_file:
        detections = json.load(json_file)

    # extract ditcionary with frames from jsonfile
    frame_dict = detections["data"][0]

    bb_centerpoint_list = []

    # calculates middle ob bb in px// to do list comprehension
    for majorkey in frame_dict:
        for subkey in frame_dict[majorkey]["classified"]:
            # converts relative x and y coordinates to px coordinates
            subkey["x_mid"] = x_res * subkey["x"]
            subkey["y_mid"] = y_res * subkey["y"]

            bb_centerpoint_list.append([subkey["x_mid"], subkey["y_mid"]])

    bb_centerpoint_arr = np.array(bb_centerpoint_list)

    return bb_centerpoint_arr


def undistort_trajectories(file):
    """function to undistort trajectories from filepath file
    load params with the load_params function
    uses numpy array of bbox center coodinates, cameramatrix and distcoefficant
    as argument for the cv.undistort function

    returns numpy array of undistorted bbox coordinates

    """

    # takes existing trajectories from dataframe und undistorts those points

    params_dict = load_params()

    bb_centerpoint_arr = load_trajectories(file)

    # load params from dictionary
    mtx = params_dict["K"]
    dist = params_dict["D"]

    # turn coefficents into matrix and array
    matrix_list = [mtx[0], mtx[1], mtx[2]]

    mtx = np.array(matrix_list)
    dist = np.array(dist)

    undistorted_points = cv.undistortPoints(bb_centerpoint_arr, mtx, dist)

    return undistorted_points


def validate_undistortion(file):
    """function to validate undistortion of trajectories
    creates to scatterplots from undistorted und distorted point set"""

    undistorted_points = undistort_trajectories(file)

    distorted_points = load_trajectories(file)

    undistored_point_list = undistorted_points.tolist()

    undistored_point_list = list(itertools.chain(*undistored_point_list))

    x, y = zip(*undistored_point_list)

    plt.scatter(x, y)
    plt.show()

    x, y = zip(*distorted_points)

    plt.scatter(x, y)
    plt.show()


# def save_trajectories():


if __name__ == "__main__":
    test_path = Path(__file__).parents[1] / "tests" / "data"
    test_path = str(test_path)

    # undistort_picture()

    # undistort_trajectories(
    #     file="H:\\06_OTCamera\\OTVision\\OTVision\\tests\\data\\testvideo_2.ottrk"
    # )

    # load_trajectories(
    #     file="H:\\06_OTCamera\\OTVision\\OTVision\\tests\\data\\testvideo_2.ottrk"
    # )

    # validate_undistortion(
    #     file="H:\\06_OTCamera\\OTVision\\OTVision\\tests\\data\\testvideo_2.ottrk"
    # )

    # get_resolution(
    #     file="H:\\06_OTCamera\\OTVision\\OTVision\\tests\\data\\testvideo_2.ottrk"
    # )
