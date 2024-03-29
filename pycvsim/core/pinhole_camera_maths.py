import numpy as np
import numbers
from typing import Tuple
from numpy.typing import NDArray
from .vector_maths import calc_closest_y_direction, rotation_matrix_to_axes


def focal_length_to_hfov(f: float, xres: int) -> float:
    """
    Calculates the horizontal field of view, in degrees, from the focal length in the x direction
        and the x resolution. (Alternatively, passing the focal length in the y direction, with the y
        resolution provides the VFOV).

    :param f: the focal length, in pixels
    :type f: float
    :param xres: the x resolution of the camera, in pixels (i.e. the width of the image_safe_zone)
    :type xres: int
    :return: the hfov, in degrees
    :rtype: float
    """
    return np.degrees(2.0 * np.arctan(xres / 2.0 / f)).item()


def hfov_to_focal_length(hfov: float, xres: int) -> float:
    """
    Calculates the x focal length, in pixels, from the hfov and the x resolution of the camera.
        (Alternatively, the y focal length can be calculated by passing the vfov and the yres)

    :param hfov: the horizontal field of view, in degrees
    :type hfov: float
    :param xres: the x resolution of the camera, in pixels (i.e. width of the image_safe_zone)
    :type xres: int
    :return: the focal length, in pixels
    :rtype: float
    """
    hfov = np.radians(hfov)
    return xres / 2.0 / np.tan(hfov / 2)


def hfov_to_vfov(hfov: float, xres: int, yres: int) -> float:
    """
    Calculates the vfov, from the corresponding hfov, the xres and yres

    :param hfov: the vertical field of view, in degrees
    :type hfov: float
    :param xres: the x resolution of the camera, in pixels (i.e. the width of the image_safe_zone)
    :type xres: int
    :param yres: the y resolution of the camera, in pixels (i.e. the height of the image_safe_zone)
    :type yres: int
    :return: the vfov, in degrees
    :rtype: float
    """
    hfov_radians = np.radians(hfov)
    vfov_radians = 2 * np.arctan(np.tan(hfov_radians / 2) * float(yres) / xres)
    return np.degrees(vfov_radians).item()


def calculate_hfov_for_safe_zone(x_res, x_res_safe, hfov):
    """
    Calculates the hfov to use to create a 'safe zone' around the image_safe_zone,
    to use when apply distortion map
    :return:
    """
    hfov_safe = 2*np.arctan(x_res_safe/x_res*np.tan(np.radians(hfov/2)))
    return np.degrees(hfov_safe).item()


def create_camera_matrix(cx: float, cy: float, fx: float, fy: float = None):
    """

    :param cx:
    :param cy:
    :param fx:
    :param fy:
    :return:
    """

    if fy is None:
        fy = fx

    camera_matrix = np.array([[fx, 0.0, cx],
                              [0.0, fy, cy],
                              [0.0, 0.0, 1.0]], dtype=np.float32)
    return camera_matrix
