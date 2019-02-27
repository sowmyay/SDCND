import func_files.DrawLane as D
import func_files.PerspectiveTransform as PT
import func_files.Threshold as T
import func_files.Undistortion as UD
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from func_files import Curvature as Cur, LaneBoundary as LB


def plotPerspectiveTransform(fname, img, thresholded, warped):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original - ' + fname, fontsize=15)
    ax2.imshow(thresholded, cmap='gray')
    ax2.set_title('Thresholded binary - ' + fname, fontsize=15)
    ax3.imshow(warped, cmap='gray')
    ax3.set_title('Warped - ' + fname, fontsize=15)


def plotFinal(result, mapped, left_fitx, right_fitx, ploty, fname):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    f.tight_layout()
    result = result.astype(int)
    ax1.imshow(result)
    ax1.set_title(fname, fontsize=50)
    ax1.plot(left_fitx, ploty, color='yellow')
    ax1.plot(right_fitx, ploty, color='yellow')

    ax2.imshow(mapped)


def combining(mtx, dist, fname, nameRemove=0, verbose=True, left_fit=None, right_fit=None):
    img = mpimg.imread(fname)
    n = len("../test_images/")
    undist = UD.cal_undistort(img, mtx, dist, fname, nameRemove=n, show=False, compareUndist=False)
    # cv2.imwrite("undist_images/" + fname[n:],cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
    thresholded = T.getThresholdImage(undist)

    img, warped, unwarped, M, M_inv = PT.perspectiveTransform(undist, thresholded)
    warped.astype('uint8')
    if verbose:
        plt.imshow(warped)
    #     if verbose:
    #         plotPerspectiveTransform(fname[nameRemove:-4], img, thresholded, warped)

    result, left_fit, right_fit, left_fitx, right_fitx = LB.search_around_poly(warped, left_fit, right_fit)

    warped = warped * 255
    img_shape = warped.shape
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    left_curverad, right_curverad = Cur.measure_curvature_real(ploty, left_fitx, right_fitx)
    center_distance = Cur.distance_from_center(img_shape, left_fitx, right_fitx)
    average_curve_rad = (left_curverad + right_curverad) / 2

    mapped = D.warp_lane(warped, img, ploty, left_fitx, right_fitx, M_inv)

    if verbose:
        print(fname[nameRemove:-4], "Curvature", average_curve_rad, 'm', "Offset", center_distance, "m")
        plotFinal(result, mapped, left_fitx, right_fitx, ploty, fname[nameRemove:-4])
