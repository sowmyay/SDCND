import numpy as np

def measure_curvature_real(ploty, leftx, rightx):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad


def distance_from_center(img_shape, left_fitx, right_fitx):
    # compute the offset from the center
    lane_center = (right_fitx[0] + left_fitx[0]) / 2
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    center_offset_pixels = abs(img_shape[0] / 2 - lane_center)
    center_offset_mtrs = xm_per_pix * center_offset_pixels
    offset_string = "Center offset: %.2f m" % center_offset_mtrs
    return center_offset_mtrs
