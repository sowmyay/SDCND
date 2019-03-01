import numpy as np
class Line():
    def __init__(self, current_fit, allx, recent_xfitted = [], bestx = None, best_fit = None, n=5):
        # x values of the last n fits of the line
        self.recent_xfitted = recent_xfitted
        # average x values of the fitted line over the last n iterations
        self.bestx = bestx
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = best_fit

        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients for the most recent fit
        self.current_fit = current_fit
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = allx
        self.n = n

    def compareToBestFit(self, threshold):
        if self.best_fit is None:
            self.best_fit = self.current_fit
            self.recent_xfitted.append(self.allx)
            self.bestx = self.allx

        else:
            diff = np.mod(self.best_fit - self.current_fit)
            m1 = np.max(diff)
            m2 = np.min(diff)

            if (m1-m2 < threshold):
                count = len(self.recent_xfitted)
                if count == self.n:
                    self.recent_xfitted = self.recent_xfitted[:, 1:]
                    count = self.n-1

                if len(self.recent_xfitted) < self.n:
                    self.best_fit = np.add(count * self.best_fit, self.current_fit) / (count + 1)
                    self.recent_xfitted.append(self.allx)
                    self.bestx = np.mean(self.recent_xfitted, axis=0)

        return self.recent_xfitted, self.best_fit, self.bestx

    def parallelCheck(self, l, threshold):
        diff = np.mod(self.bestx - l.bestx)
        m1 = np.max(diff)
        m2 = np.min(diff)

        return True if (m1 - m2 < threshold) else False

    def distanceCheck(self, l, expectedDistance, threshold):
        diff = np.mod(self.bestx - l.bestx)
        avg = np.mod(np.mean(diff))
        v = np.mod(expectedDistance - avg)
        return True if (v < threshold) else False
