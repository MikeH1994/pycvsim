import cv2
import numpy as np
import os
import copy
import pickle
from numpy.typing import NDArray


def scale_reprojection_matrix(q: NDArray, scale_factor: float):
    q = np.copy(q)
    q[:, 3] *= scale_factor
    return q


def depth_to_disparity(q: NDArray, depth: float):
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
