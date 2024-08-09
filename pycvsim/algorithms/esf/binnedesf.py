from __future__ import annotations
from pycvsim.algorithms.esf.esf import ESF, Edge
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.interpolate import interp1d
from pycvsim.algorithms.esf.esf import ESF
from pycvsim.algorithms.esf.utils import bin_data
from typing import Union
import scipy.interpolate
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


class BinnedESF(ESF):
    def __init__(self, img: NDArray, edge: Edge, ** kwargs):
        self.x0 = None
        self.x1 = None
        self.interpolation_fn: scipy.interpolate.InterpolatedUnivariateSpline = None
        self.bin_centres = []
        self.bin_values = []
        self.bin_std = []
        self.bin_range = []
        self.bins_per_pixel = kwargs["bins_per_pixel"] if "bins_per_pixel" in kwargs else 4

        super().__init__(img, edge, **kwargs)

    def fit(self, x, f, **kwargs):
        self.bin_centres, self.bin_values, self.bin_std, self.bin_range = bin_data(x, f, bins_per_pixel=self.bins_per_pixel)
        self.interpolation_fn = scipy.interpolate.InterpolatedUnivariateSpline(self.bin_centres, self.bin_values, k=1, ext=1)
        self.x0 = np.min(self.bin_centres)
        self.x1 = np.max(self.bin_centres)
        return None

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        return self.interpolation_fn(x)

    def plot(self, title=None, xlim=None, stride=1, new_figure=True, show=True):
        x_fit = np.linspace(np.min(self.bin_centres), np.max(self.bin_centres), 5000)
        y_fit = self.f(x_fit)
        if new_figure:
            plt.figure()
        if title is not None:
            plt.title(title)
        plt.scatter(self.esf_x[::stride], self.esf_f[::stride], label="data")
        plt.scatter(self.bin_centres, self.bin_values, label="bins")
        plt.plot(x_fit, y_fit, label="fit")
        plt.legend(loc=0)
        if xlim is not None:
            plt.xlim(xlim)
        if show:
            plt.show()
