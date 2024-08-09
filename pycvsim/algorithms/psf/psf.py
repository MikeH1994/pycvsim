import numpy as np
from numpy.typing import NDArray


class PSF:
    psf_kernel: NDArray

    def __init__(self):
        pass

    def deconvolve(self, img: NDArray):
        return self.deconvolution_function(img, self.psf_kernel)

    @staticmethod
    def deconvolution_function(img: NDArray, kernel: NDArray):
        pass
