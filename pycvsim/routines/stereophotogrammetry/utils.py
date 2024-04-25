import cv2
import numpy as np
import os
import copy
import pickle
from typing import Tuple
from numpy.typing import NDArray

"""
Useful sources:

(1) persective transform matrix: 
https://answers.opencv.org/question/187734/derivation-for-perspective-transformation-matrix-q/
"""

def scale_reprojection_matrix(q: NDArray, scale_factor: float):
    q = np.copy(q)
    q[:, 3] *= scale_factor
    return q


def depth_to_disparity(q: NDArray, depth: float):
    """
    For a given perspective transform matrix Q, find the disparity that corresponds a given depth.
    :param q: The
    :param depth: The distance from the camera,
    :return:
    """
    # d = cx - cx' - f Tx/Z' = a + b
    # a = cx-cx' = (cx-cx')/Tx * Tx = Q[3][3]/-Q[3][2]
    # b = -f Tx/Z'= Q[2][3]/Q[3][2]/depth

    a = q[3][3] / -q[3][2]
    b = q[2][3] / q[3][2] / depth
    return a + b


def disparity_to_depth(q: NDArray, disparity: float):
    # z' = z/w = fTx / (cx - cx' -d)
    # z = f = Q[2][3]
    # w = (cx-cx')/Tx + -d/Tx
    # -1/Tx =Q[3][2]; (cx-cx)'/Tx = Q[3][3]
    w = q[3][3] + disparity * q[3][2]
    z = q[2][3]
    return z / w


def disparity_to_position(q: NDArray, u: float, v: float, d: float):
    """



    Q =
            1       0        0       -c_x
            0       1        0       -c_y
            0       0        1       f
            0       0       -1/T_x   (c_x - c_x')/T_x
    X = u - cx = u + Q[0][3]
    y = v - cy = v + Q[1][3]
    z = f = Q[2][3]
    w = - d/Tx + (cx-cx')/Tx  = d*Q.at(3,2) + Q.at(3,3)
    posn = (X/w,y/w,z/w)
    """

    x = u + q[0][3]
    y = v + q[1][3]
    z = q[2][3]
    w = d * q[3][2] + q[3][3]
    return np.array([x / w, y / w, z / w])


def compute_min_and_max_disparity(q: NDArray, min_distance: float, max_distance: float) -> Tuple[int, int]:
    """
    Computes the min, max and number of disparities, to be used in the stereo matching process
    The min and max disparities define how far left or right the SGBM algorithm will search in image 2
    from the corresponding coordinates in image 1. This relates to the distance from the camera (the depth)
    that we are searching in. Because the disparities are required to be supplied in multiples of 16,
    the returned values do not match the requested distances exactly.

    :param q: the perpective transformation matrix Q, generated from the cv2.stereoRectify function
    :param min_distance: the minimum distance to use, in the same units as we calibrated in
    :param max_distance: the maxmimum distance to use, in the same units as we calibrated in
    :return: min_disp, max_disp
    """
    disp1 = int(depth_to_disparity(q, min_distance))
    disp2 = int(depth_to_disparity(q, max_distance))
    min_disp = min(disp1, disp2) - min(disp1, disp2) % 16
    max_disp = max(disp1, disp2) + (16 - max(disp1, disp2) % 16)
    return min_disp, max_disp
