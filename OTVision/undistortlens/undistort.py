import cv2 as cv
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import itertools

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
            dst = dst[y: y + h, x: x + w]

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


def load_trajectories(x_res=800, y_res=600):
    """ loads json file from tracking algorithm
        converts relatives coordinates to absolut coordinates in px
        calculates middle of bounding box

        returns array with points to undistort
    """

    with open(
        "H:\\06_OTCamera\\OTVision\\OTVision\\tests\\data\\testvideo_2.ottrk", "r"
    ) as json_file:
        detections = json.load(json_file)

    # extract ditcionary with frames from jsonfile
    frame_dict = detections["data"][0]

    bb_centerpoint_list = []

    # calculates middle ob bb in px
    for majorkey in frame_dict:
        for subkey in frame_dict[majorkey]["classified"]:
            subkey["x[px]"] = x_res * subkey["x"]
            subkey["y[px]"] = y_res * subkey["y"]
            subkey["x_mid"] = subkey["x[px]"] - (subkey["h"] * 0.5 * x_res)
            subkey["y_mid"] = subkey["y[px]"] + (subkey["w"] * 0.5 * y_res)

            bb_centerpoint_list.append([subkey["x_mid"], subkey["y_mid"]])

    bb_centerpoint_arr = np.array(bb_centerpoint_list)

    return bb_centerpoint_arr


def undistort_trajectories():
    # takes existing trajectories from dataframe und undistorts those points

    params_dict = load_params()

    bb_centerpoint_arr = load_trajectories()

    # load params from dictionary
    mtx = params_dict["K"]
    dist = params_dict["D"]

    # turn coefficents into matrix and array
    matrix_list = [mtx[0], mtx[1], mtx[2]]

    mtx = np.array(matrix_list)
    dist = np.array(dist)

    undistorted_points = cv.undistortPoints(bb_centerpoint_arr, mtx, dist)

    return undistorted_points

def validate_undistortion():
    """function to validate undistortion of trafectories
    """

    undistorted_points = undistort_trajectories()

    distorted_points = load_trajectories()

    list1 = undistorted_points.tolist()
 
    list1 = list(itertools.chain(*list1))

    x ,y = zip(*list1)

    plt.scatter(x ,y)
    plt.show()

    x ,y = zip(*distorted_points)

    plt.scatter(x ,y)
    plt.show()


if __name__ == "__main__":
    # undistort_picture()

    # undistort_trajectories()

    validate_undistortion()
