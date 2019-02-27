import glob

import func_files.Calibration as C
import matplotlib.image as mpimg


from func_files import Combine as Comb

mtx, dist = C.calibrateCamera(verbose=False)

test_images = glob.glob("../test_images/*")

for fname in test_images:
    img = mpimg.imread(fname)
    n = len("../test_images/")
    Comb.combining(mtx, dist, fname, nameRemove=n, verbose=True)
