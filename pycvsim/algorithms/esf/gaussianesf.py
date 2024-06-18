from __future__ import annotations
from scipy.optimize import curve_fit
from scipy.special import erf
from pycvsim.algorithms.esf.esf import ESF


class GaussianESF(ESF):
    def fit(self, x, f, **kwargs):
        n_terms = kwargs["n_terms"] if "n_terms" in kwargs else 4
        bounds_lower = self.get_bounds(n_terms, [-100, 0])
        bounds_upper = self.get_bounds(n_terms, [100, 100])
        p0 = self.get_bounds(n_terms, [0.1, 0.1])

        self.params, _ = curve_fit(self.fn, x, f, maxfev=40000, bounds=[bounds_lower, bounds_upper], p0=p0)
        return self.params

    @staticmethod
    def fn(x, *args):
        if len(args) % 2 != 0:
            raise ValueError("In ESF.fn(): the number of terms in the function must be even")
        f = 0.5
        for i in range(len(args)//2):
            a_i = args[i*2]
            b_i = args[i*2 + 1]
            f += a_i*erf(x/b_i)
        return f
