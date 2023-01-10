import ctypes
from ctypes import cdll
from numpy.ctypeslib import ndpointer
import numpy as np


class TrajectoryInterpolatorCPP:
    def __init__(self, points, tstamps, alpha=0):
        lib = cdll.LoadLibrary('./common/cpp/libmain.so')
        self.interpolator = lib.trajectory_interpolator2_cpp
        self.interpolator.restype = ndpointer(dtype=ctypes.c_double, shape=(12,), flags="C_CONTIGUOUS")
        self.interpolator.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_double]
        self.params = []
        self.ref_time = 0
        self.time_offset = 1
        self.alpha = float(alpha)
        self.create_interpolator(points, tstamps)

    def create_interpolator(self, points, tstamps):
        self.ref_time = tstamps[0]

        length = len(tstamps)
        interpolator_args = np.zeros(length*3)
        interpolator_args[:length] = tstamps
        interpolator_args[length:length*2] = points[:, 0]
        interpolator_args[length*2:length*3] = points[:, 1]

        self.params = self.interpolator(interpolator_args, length*3, self.alpha)

    def get_point(self, tstamp):
        tstamp = (tstamp - self.ref_time) / 1e9 + self.time_offset
        standardized_points = np.array([
            self.params[0] + self.params[1] * tstamp + self.params[2] * tstamp**2 + self.params[3] * tstamp**3,
            self.params[4] + self.params[5] * tstamp + self.params[6] * tstamp**2 + self.params[7] * tstamp**3
        ])
        means = np.array([self.params[8], self.params[9]])
        stds = np.array([self.params[10], self.params[11]])
        return standardized_points * stds + means