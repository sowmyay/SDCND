
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

def getImage(imgName):
    img = cv2.imread('test_images/' + imgName)
    return img

def showImages(img):
    fig = plt.figure(figsize=(2, 2))
    fig.add_subplot(2, 2, 1)
    plt.imshow(img)
    conv_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    fig.add_subplot(2, 2, 2)
    plt.imshow(conv_img1)

    conv_img2 = selectLaneColor(conv_img1)
    fig.add_subplot(2, 2, 3)
    plt.imshow(conv_img2)

    # conv_img3 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # fig.add_subplot(2, 2, 4)
    # plt.imshow(conv_img3)

    plt.show()
    return conv_img2

def selectLaneColor(img):
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ## mask of white (200, 200, 200) ~ (255, 255,255)
    # white1 = np.uint8([[[200, 200, 200]]])
    # hsv_white1 = cv2.cvtColor(white1, cv2.COLOR_RGB2HLS)
    #
    white2 = np.uint8([[[255, 255,150]]])
    hsv_white2 = cv2.cvtColor(white2, cv2.COLOR_RGB2YUV)[0][0]

    mask1 = cv2.inRange(img, np.array([138, 0, 136]), np.array([225, 75, 148]))

    ## mask of yellow (190, 190, 0) ~ (255, 255, 255)
    mask2 = cv2.inRange(img, np.array([138, 0, 136]), np.array([225, 75, 148]))

    ## final mask and masked
    mask = cv2.bitwise_or(mask1, mask2)
    target = cv2.bitwise_and(img, img, mask=mask)
    # plt.imshow(target)
    return target

test_images = os.listdir("test_images/")

for i, imgName in enumerate(test_images):
    if imgName != ".DS_Store":
        img = getImage(imgName)
        img1 = showImages(img)
        # plt.imshow(img1)
        cv2.imwrite('test_images_output/' + imgName, img1)

