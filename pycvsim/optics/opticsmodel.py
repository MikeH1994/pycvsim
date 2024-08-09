import numpy as np
from numpy.typing import NDArray


def create_airy_kernel(wavelengths):
    pass


class OpticsModel:
    def __init__(self, wavelengths: NDArray = np.array([630.0, 532.0, 467.0])):
        self.wavelengths = wavelengths


class DiffractionLimitedOptics(OpticsModel):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    pass
