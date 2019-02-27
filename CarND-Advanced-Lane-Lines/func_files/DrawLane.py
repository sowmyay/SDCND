import numpy as np
import cv2

def warp_lane(warped, img, ploty, left_fitx, right_fitx, M_inv):
    img_shape = img.shape
    img_size = (img_shape[1], img_shape[0])
    out_img = np.dstack((warped, warped, warped)) * 255

    left_line_window = np.array(np.transpose(np.vstack([left_fitx, ploty])))

    right_line_window = np.array(np.flipud(np.transpose(np.vstack([right_fitx, ploty]))))

    line_points = np.vstack((left_line_window, right_line_window))

    cv2.fillPoly(out_img, np.int_([line_points]), [0, 255, 0])

    unwarped = cv2.warpPerspective(out_img, M_inv, img_size, flags=cv2.INTER_LINEAR)
    unwarped = unwarped.astype('uint8')
    result = cv2.addWeighted(img, 1, unwarped, 0.3, 0)

    return result