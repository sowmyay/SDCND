import func_files.DrawLane as D
import func_files.PerspectiveTransform as PT
import func_files.Threshold as T
import func_files.Undistortion as UD
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from func_files.Smoothing import Line
import cv2
from func_files.Curvature import measure_curvature_real, distance_from_center
from func_files.LaneBoundary import search_around_poly


def plotPerspectiveTransform(fname, img, thresholded, warped):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original - ' + fname, fontsize=15)
    ax2.imshow(thresholded, cmap='gray')
    ax2.set_title('Thresholded binary - ' + fname, fontsize=15)
    ax3.imshow(warped, cmap='gray')
    ax3.set_title('Warped - ' + fname, fontsize=15)


def plotFinal(original, result, mapped, left_fitx, right_fitx, ploty, fname):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 9))
    f.tight_layout()
    result = result.astype(int)
    ax1.imshow(original)
    ax1.set_title(fname, fontsize=50)
    ax2.imshow(result)
    ax2.plot(left_fitx, ploty, color='yellow')
    ax2.plot(right_fitx, ploty, color='yellow')
    ax3.imshow(mapped)


def final_pipeline(mtx, dist, img=None, fname=None, nameRemove=0, verbose=False, best_left={}, best_right = {}):
    if img is None:
        img = mpimg.imread(fname)

    undist = UD.cal_undistort(img, mtx, dist, fname, nameRemove=nameRemove, verbose=False, compareUndist=False)
    # cv2.imwrite("undist_images/" + fname[n:],cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
    thresholded = T.getThresholdImage(undist)
    #     plt.imshow(thresholded)

    image, warped, unwarped, M, M_inv = PT.perspectiveTransform(undist, thresholded)
    warped.astype('uint8')

    if verbose:
        #         plt.imshow(warped)
        plotPerspectiveTransform(fname[nameRemove:-4], img, thresholded, warped)

    result, best_left, best_right = search_around_poly(warped, best_left=best_left, best_right = best_right)

    left_fit = best_left["best_fit"]
    right_fit = best_right["best_fit"]

    left_fitx = best_left["bestx"]
    right_fitx = best_right["bestx"]

    warped = warped * 255
    img_shape = warped.shape
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    left_curverad, right_curverad = measure_curvature_real(ploty, left_fitx, right_fitx)
    offset = distance_from_center(img_shape, left_fitx, right_fitx)
    average_curve_rad = (left_curverad + right_curverad) / 2

    mapped = D.warp_lane(warped, undist, ploty, left_fitx, right_fitx, M_inv)

    curv_str = "Curvature: %.2f m" % average_curve_rad
    offset_str = "Center offset: %.2f m" % offset

    cv2.putText(mapped, curv_str, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), thickness=2)
    cv2.putText(mapped, offset_str, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), thickness=2)

    if verbose:
        #         print(fname[nameRemove:-4], "Curvature", average_curve_rad, 'm', "Offset", offset, "m")
        plotFinal(img, result, mapped, left_fitx, right_fitx, ploty, fname[nameRemove:-4])

    return mapped, best_left, best_right
