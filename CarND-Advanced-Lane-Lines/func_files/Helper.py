import math
import numpy as np
import cv2
from func_files.Threshold import region_of_interest

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def getIntercepts(lines, min_y, max_y, color=[255, 0, 0], thickness=10):
    l = []
    if lines is not None:
        slope_left, intercept_left, slope_right, intercept_right = avgLines(lines=lines)

        if slope_left != 0:
            x11 = int((min_y - intercept_left) / slope_left)
            x12 = int((max_y - intercept_left) / slope_left)
            l += [(x11, min_y, x12, max_y)]

        if slope_right != 0:
            x21 = int((min_y - intercept_right) / slope_right)
            x22 = int((max_y - intercept_right) / slope_right)
            l += [(x21, min_y, x22, max_y)]
    return l


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, min_y, max_y):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    l = getIntercepts(lines, min_y=min_y, max_y=max_y)
    return l


def avgLines(lines):
    left_slope = 0
    left_weights = 0
    left_intercept = 0
    right_slope = 0
    right_weights = 0
    right_intercept = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1 and abs(slope) < 0.05:
                continue  # ignore a vertical line
            slope = (y2 - y1) * 1.0 / (x2 - x1)
            intercept = y1 - slope * x1
            weight = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:
                left_slope += slope * weight
                left_intercept += intercept * weight
                left_weights += weight
            else:
                right_slope += slope * weight
                right_intercept += intercept * weight
                right_weights += weight

    if left_weights != 0:
        left_slope = left_slope / left_weights
        left_intercept = left_intercept / left_weights
    if right_weights != 0:
        right_intercept = right_intercept / right_weights
        right_slope = right_slope / right_weights

    return left_slope, left_intercept, right_slope, right_intercept


def selectLaneColor(img):
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ## mask of white (200, 200, 200) ~ (255, 255,255)
    mask1 = cv2.inRange(img, np.array([0, 200, 0]), np.array([255, 255, 255]))

    ## mask of yellow (190, 190, 0) ~ (255, 255, 255)
    mask2 = cv2.inRange(img, np.array([10, 0, 100]), np.array([40, 255, 255]))

    ## final mask and masked
    mask = cv2.bitwise_or(mask1, mask2)
    target = cv2.bitwise_and(img, img, mask=mask)
    # plt.imshow(target)
    return target


def detectLanes(img):
    img.astype('uint8')
    conv_img1 = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
    masked_img = selectLaneColor(conv_img1)
    conv_img2 = cv2.cvtColor(masked_img, cv2.COLOR_HLS2RGB)
    conv_img3 = cv2.cvtColor(conv_img2, cv2.COLOR_RGB2GRAY)
    kernel_size = 7  # Must be an odd number (3, 5, 7...)
    blur_gray = gaussian_blur(conv_img3, kernel_size)
    canny_img = canny(blur_gray, 50, 150)

    imshape = img.shape
    vertices = np.array([[(50, imshape[0]),
                          ((imshape[1] - 100) / 2, (imshape[0] + 100) / 2),
                          ((imshape[1] + 100) / 2, (imshape[0] + 100) / 2),
                          (imshape[1] - 50, imshape[0])]],
                        dtype=np.int32)
    v = vertices[0]
    x = [v[0][0], v[1][0], v[2][0], v[3][0]]
    y = [v[0][1], v[1][1], v[2][1], v[3][1]]
    masked_edges = region_of_interest(canny_img, vertices=vertices)

    rho = 1
    theta = np.pi / 180
    threshold = 30
    min_line_length = 10
    max_line_gap = 10
    lines = hough_lines(img=masked_edges, rho=rho, theta=theta, threshold=threshold, min_line_len=min_line_length,
                        max_line_gap=max_line_gap, min_y=int((imshape[0] + 200) / 2), max_y=imshape[0])

    return lines