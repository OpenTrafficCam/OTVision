import cv2 as cv
import glob
import json 
import numpy as np

#H:\06_OTCamera\OTCamera\OTCamera\data.txt

path = "./OTVision"


# text_files = glob.glob(path + "/**/data.txt", recursive = True)


# cameraparams = text_files[0] 


# loads dictionary with parameters from 
def load_params():
    """Loads data.txt - file with the distinct camera parameters K and D
    
    K is the cameramatrix
    D are distance coefficent
    funtion returns dictionary
    """
    text_files = glob.glob(path+"/**/data.txt", recursive = True)

    cameraparams = text_files[0] 

    with open(cameraparams) as fh:

        data = fh.read()

        params_dict = json.loads(data)

    return params_dict

# function to undistort pictures 
def undistort_picture():
    """ takes params and undistorts a list of images"""

    params_dict = load_params()

    print(params_dict)

    # load params from dictionary
    mtx = params_dict["K"]
    dist = params_dict["D"]

    # turn coefficents into matrix and array
    matrix_list = [mtx[0],mtx[1],mtx[2]]

    mtx = np.array(matrix_list)
    dist = np.array(dist)

    #print(mtx, dist)

    image_list = glob.glob(path + "/**/*.jpg", recursive = True)

    i = 0

    # undistort list of images with camera coefficents
    for img in image_list:

        img = cv.imread(img)

        h,  w = img.shape[:2]

        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # undistort

        dst = cv.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        # saves undistorted pictures to path 
        cv.imwrite("H:\\06_OTCamera\OTCamera\OTCamera\imagefolder/preview{0}ud.jpg".format(str(i)), dst)

        i += 1



# def undistort_video():


if __name__ == "__main__":
    undistort_picture()



# def undistort_trajectories():



#undistort_picture()


#cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#function to undistort trajektorien



