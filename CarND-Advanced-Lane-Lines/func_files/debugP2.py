import glob
import cv2
from func_files.Calibration import calibrateCamera
import matplotlib.image as mpimg
from func_files.Combine import final_pipeline

mtx, dist = calibrateCamera(verbose=False)

test_images = glob.glob("test_images/*")
best_left={}
best_right = {}
best_left["recent_xfitted"] = []
best_left["best_fit"] = None
best_left["bestx"] = None
best_right["recent_xfitted"] = []
best_right["best_fit"] = None
best_right["bestx"] = None

for fname in test_images:
    img = mpimg.imread(fname)
    n = len("test_images/")

    mapped, best_left, best_right = final_pipeline(mtx, dist, fname=fname, nameRemove=n, verbose=False,
                                                        best_left=best_left, best_right=best_right)

    cv2.imwrite("smoothed_images/" + fname[n:], mapped)

