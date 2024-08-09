import numpy as np
from pycvsim.algorithms.esf.binnedesf import BinnedESF
from pycvsim.algorithms.lsf.lsf import LSF
import scipy.interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
from numpy.typing import NDArray


class BinnedLSF(LSF):
    def __init__(self, esf: BinnedESF):
        super().__init__(esf)
        self.x0 = esf.x0
        self.x1 = esf.x1
        self.bin_centres = esf.bin_centres
        self.bin_values = esf.interpolation_fn.derivative()(self.bin_centres)
        self.interpolation_fn = InterpolatedUnivariateSpline(self.bin_centres, self.bin_values, k=1, ext=1)
        self.bins_per_pixel = esf.bins_per_pixel

    def f(self, x: NDArray) -> NDArray:
        return self.interpolation_fn(x)
