import cv2 as cv
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pathlib import Path

# H:\06_OTCamera\OTCamera\OTCamera\data.txt

imagepath = "OTVision\\imagefolder"


parameterpath = "OTVision\\undistortlens\\data.txt"


# loads dictionary with parameters from


def load_params(parameter_textfile=parameterpath):
    """Loads data.txt - file with the distinct camera parameters K and D

    K is the cameramatrix
    D are distance coefficent
    funtion returns dictionary
    """
    text_files = glob.glob(parameter_textfile, recursive=True)

    cameraparams = text_files[0]

    with open(cameraparams) as fh:

        data = fh.read()

        params_dict = json.loads(data)

    return params_dict


# function to undistort pictures
def undistort_picture(imagefolder=imagepath):
    """ takes params and undistorts a list of images"""

    # loads dictionary
    params_dict = load_params()

    # load params from dictionary
    mtx = params_dict["K"]
    dist = params_dict["D"]

    # turn coefficents into matrix and array
    matrix_list = [mtx[0], mtx[1], mtx[2]]

    mtx = np.array(matrix_list)
    dist = np.array(dist)

    # print(mtx, dist)

    image_list = glob.glob(imagefolder + "\\*.jpg", recursive=True)

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
                "OTVision\\imagefolder_undistorted\\preview{0}_undistorted.jpg".format(
                    str(i)
                ),
                dst,
            )

            i += 1
    except KeyError:
        print("no pictures to undistort")


def get_resolution(trackingfilepath, trackingfile):
    """loads json file from tracking algorithm
    if x and y are normalized function returns height and width as integer
    if not, xy-resolution is 1
    this is necessary to calculate the absolute xy coordinates

    takes filepath of tracking data

    returns resolution as height and with or 1
    """

    with open(trackingfilepath + trackingfile, "r") as json_file:
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


def load_trajectories(trackingfilepath, trackingfile):
    """loads json file from tracking algorithm
    converts relatives coordinates to absolut coordinates
    in px by using a given resolution
    appends and converts a list of xy corrdinates for the cv.undistortpointsfunction

    returns array with points to undistort
    """
    x_res, y_res = get_resolution(trackingfilepath, trackingfile)

    with open(trackingfilepath + trackingfile, "r") as json_file:
        detections = json.load(json_file)

    # extract ditcionary with frames from jsonfile
    frame_dict = detections["data"][0]

    bb_centerpoint_list = []

    # calculates middle ob bb in px// to do list comprehension
    for majorkey in frame_dict:
        for subkey in frame_dict[majorkey]["classified"]:
            # converts relative x and y coordinates to px coordinates
            if x_res != 1 & y_res != 1:
                subkey["x_abs"] = x_res * subkey["x"]
                subkey["y_abs"] = y_res * subkey["y"]

                bb_centerpoint_list.append([subkey["x_abs"], subkey["y_abs"]])
            else:
                bb_centerpoint_list.append([subkey["x"], subkey["y"]])

    bb_centerpoint_arr = np.array(bb_centerpoint_list)

    dump_path = trackingfilepath + "undistorted_" + trackingfile

    to_json = json.dumps(frame_dict, indent=4)

    x = open(dump_path, "w")
    x.write(to_json)

    return bb_centerpoint_arr


def undistort_trajectories(trackingfilepath, trackingfile):
    """function to undistort trajectories from filepath file
    load params with the load_params function
    uses numpy array of bbox center coodinates, cameramatrix and distcoefficant
    as argument for the cv.undistort function

    returns numpy array of undistorted bbox coordinates

    """

    # takes existing trajectories from dataframe und undistorts those points

    params_dict = load_params()

    bb_centerpoint_arr = load_trajectories(trackingfilepath, trackingfile)

    # load params from dictionary
    mtx = params_dict["K"]
    dist = params_dict["D"]

    # turn coefficents into matrix and array
    matrix_list = [mtx[0], mtx[1], mtx[2]]

    mtx = np.array(matrix_list)
    dist = np.array(dist)

    undistorted_points = cv.undistortPoints(bb_centerpoint_arr, mtx, dist)

    return undistorted_points


def validate_undistortion(trackingfilepath, trackingfile):
    """function to validate undistortion of trajectories
    creates to scatterplots from undistorted und distorted point set"""

    undistorted_points = undistort_trajectories(trackingfilepath, trackingfile)

    distorted_points = load_trajectories(trackingfilepath, trackingfile)

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
    # PROBLEM: DRUCKT EIN OTVISION ZU VIEL!!
    test_path = Path(__file__).parents[1] / "tests" / "data"
    test_path = str(test_path)

    test_path = "tests\\data\\"

    testfile = "testvideo_2.ottrk"

    # undistort_picture()

    # undistort_trajectories(
    #     trackingfilepath="H:\\06_OTCamera\\OTVision\\OTVision\\tests\\data\\testvideo_2.ottrk"
    # )

    # load_trajectories(
    #     trackingfilepath="H:\\06_OTCamera\\OTVision\\OTVision\\tests\\data\\testvideo_2.ottrk"
    # )

    validate_undistortion(test_path, testfile)

    # get_resolution(
    #     trackingfilepath="H:\\06_OTCamera\\OTVision\\OTVision\\tests\\data\\testvideo_2.ottrk"
    # )
