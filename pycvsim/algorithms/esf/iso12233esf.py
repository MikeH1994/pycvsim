from __future__ import annotations
from pycvsim.algorithms.esf.esf import ESF, Edge
from scipy.optimize import curve_fit
from scipy.special import erf
from pycvsim.algorithms.esf.esf import ESF
from typing import Union
from numpy.typing import NDArray

class ISO12233ESF(ESF):
    def __init__(self, img: NDArray, edge: Edge, ** kwargs):
        super().__init__(img, edge, **kwargs)


    def fit(self, x, f, **kwargs):

        return self.params

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        return