import cv2
import matplotlib.pyplot as plt

def cal_undistort(img, mtx, dist, fname, nameRemove = 0, verbose=False, compareUndist=True):
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix = mtx,
                                                      distCoeffs = dist,imageSize = (w,h),
                                                      alpha = 1, newImgSize =(w,h))
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    dst1 = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x,y,w,h = roi
    dst1 = dst1[y:y+h, x:x+w]
    if verbose:
        if compareUndist:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
            ax3.imshow(dst)
            ax3.set_title('Undistorted - without modified camera matrix - '+ fname[nameRemove:-4], fontsize=15)
        else:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        f.subplots_adjust(hspace = .2, wspace=.05)
        ax1.imshow(img)
        ax1.set_title('Original - ' + fname[nameRemove:-4], fontsize=15)
        ax2.imshow(dst1)
        ax2.set_title('Undistorted - with modified camera matrix - '+ fname[nameRemove:-4], fontsize=15)
    return dst1
