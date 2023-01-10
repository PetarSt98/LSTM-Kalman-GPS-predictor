import numpy as np
import scipy.optimize as o
from numpy.polynomial import Polynomial


class TrajectoryInterpolatorInference:
    def __init__(self, wps, wps_tstamps):
        self.time_offset = 1
        self.ref_tstamp = wps_tstamps[0]

        self.wps_mean = wps.mean(axis=0)
        self.wps_std = wps.std(axis=0)
        wps = (wps - self.wps_mean) / self.wps_std

        self.times = self._tstamps_to_times(wps_tstamps)
        num_params = 8

        self.y = np.concatenate([wps[:, 0], wps[:, 1]])

        p0 = np.random.normal(0, 1, size=num_params)  # initial guess for lambda, mu, and sigma

        self.params = o.minimize(self.cost, p0).x

    def _tstamps_to_times(self, tstamps):
        return (np.array(tstamps) - self.ref_tstamp) / 1e9 + self.time_offset

    def get_point(self, tstamp):
        t = self._tstamps_to_times(tstamp)
        return self.get_pred(self.params, np.array(t)[None]) * self.wps_std + self.wps_mean

    @staticmethod
    def get_pred(params, times):
        x0, x1, x2, x3, y0, y1, y2, y3 = params

        Px = Polynomial([x0, x1, x2, x3])
        Py = Polynomial([y0, y1, y2, y3])
        return np.concatenate([Px(times), Py(times)])

    def cost(self, params):
        pred_y = self.get_pred(params, self.times)

        reg = (params ** 2).sum()
        reg_weight = 0

        return np.mean((pred_y - self.y) ** 2) + reg * reg_weight

    @staticmethod
    def func(t, x3, x2, x1, x0, y3, y2, y1, y0):
        from numpy.polynomial import Polynomial

        Px = Polynomial([x3, x2, x1, x0])
        Py = Polynomial([y3, y2, y1, y0])
        return np.concatenate([Px(t), Py(t)])
