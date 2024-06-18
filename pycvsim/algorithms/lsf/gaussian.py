from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit
from ssecorrection.lsf.lsf import LSF
from ssecorrection.esf import GaussianESF


class GaussianLSF(LSF):
    def __init__(self, esf: GaussianESF):
        super().__init__(esf)

    @staticmethod
    def fn(x, *args):
        if len(args) % 2 != 0:
            raise ValueError("In LSF.fn(): the number of terms in the function must be even")
        f = 0.0
        for i in range(len(args)//2):
            a_i = args[i*2]
            b_i = args[i*2 + 1]
            f += 2.0 / np.sqrt(np.pi) * a_i / b_i * np.exp(-x**2 / b_i **2)
        return f
