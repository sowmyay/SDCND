#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob


def cal_undistort(img, objpoints, imgpoints):
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist



images = glob.glob("camera_cal/calibration*.jpg")
# Array to store object and image points from all the images
objpoints = []
imgpoints = []
imgs = []
m = 9
n = 6
objp = np.zeros((m * n, 3), np.float32)
objp[:, :2] = np.mgrid[0:n, 0:m].T.reshape(-1, 2)
# print(objp)
fig = plt.figure(figsize=(16, 16))
columns = 4
rows = 5

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
        imgs.append(img)
        img = cv2.drawChessboardCorners(img, (m, n), corners, ret)
        ax1 = fig.add_subplot(rows, columns, i)
        ax1.title.set_text(fname)
        plt.imshow(img)
        # print(objpoints)
        # print(imgpoints)

for i, img in enumerate(imgs):
    obj = objpoints[i]
    print(obj)
    imgp = imgpoints[i]

    undistorted = cal_undistort(img, obj, imgpoints)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)