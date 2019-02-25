# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # COLOR_RGB2GRAY
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, min_y, max_y, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    slope_left, intercept_left, slope_right, intercept_right = avgLines(lines=lines)


    x11 = int((min_y - intercept_left)/slope_left)

    x12 = int((max_y - intercept_left) / slope_left)

    cv2.line(img, (x11, min_y), (x12, max_y), color, thickness)

    x21 = int((min_y - intercept_right) / slope_right)

    x22 = int((max_y - intercept_right) / slope_right)

    cv2.line(img, (x21, min_y), (x22, max_y), color, thickness)

    # for line in new_lines:
    #     for x1, y1, x2, y2 in line:
    #         # slope = abs((y2-y1)/(x2-x1))
    #         # if not slope < 0.05:
    #         cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, min_y, max_y):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, min_y=min_y, max_y=max_y)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
def getImage(imgName):
    img = cv2.imread('test_images/' + imgName)
    return img


def selectLaneColor(img):
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ## mask of white (200, 200, 200) ~ (255, 255,255)
    # white1 = np.uint8([[[200, 200, 200]]])
    # hsv_white1 = cv2.cvtColor(white1, cv2.COLOR_RGB2HLS)
    #
    # white2 = np.uint8([[[255, 255,255]]])
    # hsv_white2 = cv2.cvtColor(white1, cv2.COLOR_RGB2HLS)

    mask1 = cv2.inRange(img, np.array([0, 200, 0]), np.array([255, 255,255]))

    ## mask of yellow (190, 190, 0) ~ (255, 255, 255)
    mask2 = cv2.inRange(img, np.array([10,   0, 100]), np.array([40, 255, 255]))

    ## final mask and masked
    mask = cv2.bitwise_or(mask1, mask2)
    target = cv2.bitwise_and(img, img, mask=mask)
    # plt.imshow(target)
    return target

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
            slope = (y2 - y1)*1.0 / (x2 - x1)
            intercept = y1 - slope * x1
            weight = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0:
                left_slope += slope*weight
                left_intercept += intercept*weight
                left_weights += weight
            else:
                right_slope += slope * weight
                right_intercept += intercept * weight
                right_weights += weight
    avg_slope_left = left_slope/left_weights
    avg_intercept_left = left_intercept/left_weights
    avg_intercept_right = right_intercept/right_weights
    avg_slope_right = right_slope/right_weights

    return avg_slope_left, avg_intercept_left, avg_slope_right, avg_intercept_right

def detectLanes(img):
    fig = plt.figure(figsize=(2, 2))

    # gray_img = grayscale(img)
    conv_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)
    # conv_img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
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

    # canny_img1 = h.canny(blur_gray, 50, 150)

    fig.add_subplot(2, 2, 1)
    plt.imshow(conv_img1)

    fig.add_subplot(2, 2, 2)
    plt.imshow(masked_img, cmap='Greys_r')

    fig.add_subplot(2, 2, 3)
    plt.imshow(canny_img, cmap='Greys_r')

    v = vertices[0]
    x = [v[0][0], v[1][0], v[2][0], v[3][0]]
    y = [v[0][1], v[1][1], v[2][1], v[3][1]]
    plt.plot(x, y, 'b--', lw=4)

    masked_edges = region_of_interest(canny_img, vertices=vertices)

    rho = 1
    theta = np.pi / 180
    threshold = 30
    min_line_length = 10
    max_line_gap = 10
    line_image = hough_lines(img=masked_edges, rho=rho, theta=theta, threshold=threshold, min_line_len=min_line_length,
                             max_line_gap=max_line_gap, min_y=int((imshape[0] + 200) / 2), max_y=imshape[0])

    # Draw the lines on the edge image
    combo = weighted_img(img=line_image, initial_img=img)

    fig.add_subplot(2, 2, 4)
    plt.imshow(combo)
    plt.show()
    return combo

test_images = os.listdir("test_images/")

for i, imgName in enumerate(test_images):
    if imgName != ".DS_Store":
        img = getImage(imgName)
        img1 = detectLanes(img)
        # plt.imshow(img1)
        cv2.imwrite('test_images_output/' + imgName, img1)

