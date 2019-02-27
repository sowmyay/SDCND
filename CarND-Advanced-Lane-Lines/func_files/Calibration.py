import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle

#Calibration

def calibrateCamera(verbose=True):
    # Array to store object and image points from all the images
    images = glob.glob("../camera_cal/calibration*.jpg")
    objpoints = []
    imgpoints = []
    m = 9
    n = 6
    objp = np.zeros((m * n, 3), np.float32)
    objp[:, :2] = np.mgrid[0:m, 0:n].T.reshape(-1, 2)

    for i, fname in enumerate(images):
        img = mpimg.imread(fname)
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the corners of chess board
        ret, corners = cv2.findChessboardCorners(gray, (m, n), None)

        # if corners are found, add points to object and img points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            img = cv2.drawChessboardCorners(img, (m, n), corners, ret)
            if verbose:
                f, ax1 = plt.subplots(1, 1, figsize=(20, 10))
                f.subplots_adjust(hspace=.2, wspace=.05)
                ax1.imshow(img)
                ax1.set_title(fname[11:], fontsize=30)
                plt.show()

    # Do camera calibration given object points and image points
    img = cv2.imread('../camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("calibration.p", "wb"))

    return mtx, dist