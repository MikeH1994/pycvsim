import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from pycvsim.algorithms.lsf import LSF
from pycvsim.algorithms.lsf.binnedlsf import BinnedLSF
from numpy.fft import fft


def calculate_mtf(lsf: BinnedLSF):
    values = lsf.bin_values
    mtf = np.abs(fft(values))
    mtf /= mtf[0]

    frequencies = np.arange(values.shape[0])/(np.cos(np.radians(lsf.angle % 90.0))*lsf.bins_per_pixel*values.shape[0])
    frequencies -= frequencies[0]
    return frequencies, mtf
