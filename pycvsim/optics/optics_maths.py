import numpy as np
from numpy.typing import NDArray
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.special


def airy_disk_fn(x, y, radius, x_0 = 0.0, y_0 = 0.0):
    """
    Returns the values for a 2D airy disk function
    :param radius: the radius of the airy disk
    :return:


    https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.AiryDisk2D.html
    https://github.com/astropy/astropy/blob/f788114c8a01a966cebdc8710674f88f9f1cdb7c/astropy/modeling/functional_models.py#L2909
    """

    rz = scipy.special.jn_zeros(1, 1)[0] / np.pi
    r = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2) / (radius / rz)
    z = np.ones(r.shape)
    rt = np.pi * r[r > 0]
    z[r > 0] = (2.0 * scipy.special.j1(rt) / rt) ** 2

    return z


def generate_airy_kernel(wavelength, focal_length, xres, hfov, aperture_diameter):
    """
    https://github.com/astropy/astropy/blob/f788114c8a01a966cebdc8710674f88f9f1cdb7c/astropy/convolution/kernels.py#L713
    https://github.com/astropy/astropy/blob/f788114c8a01a966cebdc8710674f88f9f1cdb7c/astropy/modeling/functional_models.py#L2909

    :param wavelengths:
    :param focal_length:
    :return:
    """


def calculate_airy_radius_in_pixels(wavelength, hfov, xres, aperture_diameter):
    """
    Calculate the radius of the airy disk (the location of the first minima)
    in terms of pixels

    :param wavelength:
    :param focal_length:
    :param aperture_diameter:
    :return:

    """
    #    the first minima of the airy disk in an optical system occurs at sin(theta) = rz*lambda / D
    # D = aperture diameter, rz = 1.219669...
    # i.e. theta = asin(rz*lambda/D)
    rz = scipy.special.jn_zeros(1, 1)[0] / np.pi  # rz = 1.219669...
    theta = np.asin(rz*wavelength/aperture_diameter) # angular radius of minima
    radius_px = (xres / 2) * np.tan(theta)/np.tan(hfov/2) # radius in pixels onscreen
    return radius_px

def create_airy_kernel(wavelengths, focal_length: float, aperture_diameter: float):
    f_number = focal_length / aperture_diameter
