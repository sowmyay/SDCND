import cv2
import numpy as np

from func_files.Helper import detectLanes


def perspectiveTransform(img, thresholded):
    # Vertices extracted manually for performing a perspective transform

    lines = detectLanes(img)
    if len(lines) > 1:
        left_line = lines[0]
        right_line = lines[1]

        bottom_left = left_line[2:]
        bottom_right = right_line[2:]
        top_left = left_line[:2]
        top_right = right_line[:2]
    else:
        bottom_left = [160, 720]
        bottom_right = [1250, 720]
        top_left = [600, 400]
        top_right = [700, 400]
    h, w, b = img.shape
    source = np.float32([bottom_left, bottom_right, top_right, top_left])
    pts = np.array([bottom_left, bottom_right, top_right, top_left], np.int32)
    pts = pts.reshape((-1, 1, 2))
    copy = img.copy()
    cv2.polylines(copy, [pts], True, (255, 0, 0), thickness=5)

    bottom_left = [320, 720]
    bottom_right = [920, 720]
    top_left = [320, 1]
    top_right = [920, 1]

    dst = np.float32([bottom_left, bottom_right, top_right, top_left])
    M = cv2.getPerspectiveTransform(source, dst)
    M_inv = cv2.getPerspectiveTransform(dst, source)
    image_shape = img.shape
    img_size = (image_shape[1], image_shape[0])

    warped = cv2.warpPerspective(thresholded, M, img_size, flags=cv2.INTER_LINEAR)

    unwarped = cv2.warpPerspective(warped, M_inv, img_size, flags=cv2.INTER_LINEAR)

    return copy, warped, unwarped, M, M_inv