from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ssecorrection.esf import ESF


class LSF:
    esf: ESF
    params: NDArray
    angle: float

    def __init__(self, esf: ESF):
        self.esf = esf
        self.params = esf.params
        self.angle = esf.angle

    def f(self, x: NDArray) -> NDArray:
        return self.fn(x, *self.params)

    @staticmethod
    def fn(x: NDArray, *args):
        raise Exception("Base function LSF.f() called")
