import numpy as np
import cv2

def hls_s_layer(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    return hls


def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255):
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1 if orient == 'x' else 0, 0 if orient == 'x' else 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output


def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return binary_output


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.sqrt(sobelx ** 2)
    abs_sobely = np.sqrt(sobely ** 2)
    grad = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(grad)
    binary_output[(grad > thresh[0]) & (grad < thresh[1])] = 1
    return binary_output


def region_of_interest(img, vertices):
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


def getThresholdImage(image, verbose=False):
    hls = hls_s_layer(image)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    H_thresh = (10, 40)
    H_binary = np.zeros_like(H)
    H_binary[(H > H_thresh[0]) & (H <= H_thresh[1])] = 1

    S_thresh = (100, 255)
    S_binary = np.zeros_like(S)
    S_binary[(S > S_thresh[0]) & (S <= S_thresh[1])] = 1

    L_thresh = (200, 255)
    L_binary = np.zeros_like(L)
    L_binary[(L > L_thresh[0]) & (L <= L_thresh[1])] = 1

    thresh = (150, 255)
    R = image[:, :, 0]
    R_binary = np.zeros_like(R)
    R_binary[(R > thresh[0]) & (R <= thresh[1])] = 1

    mag_binary = mag_thresh(S, sobel_kernel=3, mag_thresh=(50, 200))
    dir_binary1 = dir_threshold(L_binary, sobel_kernel=3, thresh=(np.pi / 6, np.pi / 2))
    dir_binary2 = dir_threshold(R_binary, sobel_kernel=3, thresh=(np.pi / 6, np.pi / 2))
    dir_binary3 = dir_threshold(S_binary, sobel_kernel=3, thresh=(np.pi / 6, np.pi / 2))
    dir_binary = (((dir_binary1 == 1) | ((dir_binary2 == 1) & (dir_binary3 == 1)))
                  | ((dir_binary2 == 1) & ((dir_binary1 == 1) | (dir_binary3 == 1)))
                  | ((dir_binary3 == 1) | ((dir_binary1 == 1) & (dir_binary2 == 1))))

    combined = np.zeros_like(R)

    gradx = abs_sobel_thresh(S, orient='x', thresh_min=10, thresh_max=100)
    grady = abs_sobel_thresh(S_binary, orient='y', thresh_min=5, thresh_max=200)

    #     combined[((gradx == 1) | ((mag_binary == 1) | (dir_binary == 1)) & (L_binary == 1)) | (SSeq_binary == 1)] = 1
    combined[((mag_binary == 1) | (dir_binary == 1)) | ((R_binary == 1) & (S_binary == 1)) | (L_binary == 1)] = 1
    #     combined[(gradx == 1)] = 1
    h, w = combined.shape
    vertices = np.array([[(250, h - 20),
                          ((w - 120) / 2, (h + 160) / 2),
                          ((w + 200) / 2, (h + 160) / 2),
                          (w - 120, h - 20)]],
                        dtype=np.int32)

    v = vertices[0]
    x = [v[0][0], v[1][0], v[2][0], v[3][0]]
    y = [v[0][1], v[1][1], v[2][1], v[3][1]]

    masked_image = region_of_interest(combined, vertices)

    return masked_image