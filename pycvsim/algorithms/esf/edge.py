from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from pycvsim.algorithms.esf.utils import get_edge_from_image
from scipy.special import expit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import InterpolatedUnivariateSpline
from typing import Union, List, Tuple
import skimage.filters.thresholding
from pycvsim.core.image_utils import convert_to_8_bit


class Edge:
    """

    """
    def __init__(self, p0: NDArray, p1: NDArray):
        self.x0, self.y0 = p0
        self.x1, self.y1 = p1

        self.inf_gradient = self.x1 == self.x0
        self.m = (self.y1 - self.y0) / (self.x1 - self.x0) if not self.inf_gradient else np.inf
        self.c = self.y0 - self.x0*self.m
        self.angle = self.gradient_to_angle(self.m)
        self.is_vertical = self.gradient_is_vertical(self.m)

        if self.is_vertical:
            if self.y0 > self.y1:
                self.x0, self.y0, self.x1, self.y1 = self.x1, self.y1, self.x0, self.y0
        else:
            if self.x0 > self.x1:
                self.x0, self.y0, self.x1, self.y1 = self.x1, self.y1, self.x0, self.y0

    def get_bounds(self, min_x: float = None, min_y: float = None, max_x: float = None, max_y: float = None,
                   return_as_int: bool = False):
        x0, y0, x1, y1 = self.x0, self.y0, self.x1, self.y1
        if self.is_vertical:
            assert(y0 <= y1)
            if min_y is not None and y0 < min_y:
                y0 = min_y
                x0 = self.get_edge_x(self.y0)
            if max_y is not None and y1 > max_y:
                y1 = max_y
                x1 = self.get_edge_x(self.y1)
        else:
            assert(x0 <= x1)
            if min_x is not None and x0 < min_x:
                x0 = 0.0
                y0 = self.get_edge_y(x0)
            if max_x is not None and x1 > max_x:
                x1 = max_x
                y1 = self.get_edge_y(x1)
        if return_as_int:
            return int(x0), int(y0), int(x1), int(y1)
        return x0, y0, x1, y1

    def distance_to_edge(self, x: float, y: float):
        """
        for a line of the form Ax + By + c = 0,
        the distance between the line and point x0,y0 is
        (Ax0 + By0 +c)/sqrt(m^2 + 1)
        given the defined line f=mx+c, this gives the eqn of f-mx-c = 0.
        plugging f-mx-c in to the equation for the line gives
        (f - mx - c) / (sqrt(m^2 + 1)

        :param x:
        :param y:
        :return:
        """
        if self.inf_gradient:
            return x - self.x0
        else:
            return (y - (self.m * x + self.c)) / np.sqrt(self.m ** 2 + 1)

    # noinspection PyMethodMayBeStatic
    def gradient_to_angle(self, m: float):
        return np.degrees(np.arctan(m)) % 360.0

    # noinspection PyMethodMayBeStatic
    def gradient_is_vertical(self, m: Union[float, None]):
        return np.abs(m) > 1 if m is not None else True

    def get_edge_x(self, y):
        # y = mx + c, therefore, x = (y - c)/m
        return (y - self.c) / self.m if not self.inf_gradient else self.x0

    def get_edge_y(self, x):
        return self.m * x + self.c

    def point_above_line(self, x, y):
        # if line is vertical
        if self.inf_gradient:
            return x > self.x0
        else:
            y_line = self.get_edge_y(x)
            return y < y_line

    def draw(self, image: NDArray, title=None, xlim=None, new_figure=True, show=True):
        if new_figure:
            plt.figure()
        if title is not None:
            plt.title(title)
        image = convert_to_8_bit(image)
        plt.imshow(image)
        plt.plot([self.x0, self.x1], [self.y0, self.y1], color='r')
        if show:
            plt.show()

    @staticmethod
    def create_from_image(image: NDArray, edge_detection_mode="lsf", display=False):
        p0, p1, _, _ = get_edge_from_image(image, edge_detection_mode=edge_detection_mode, display=display)
        return Edge(p0, p1)

    @staticmethod
    def create_from_two_points(p0: NDArray, p1: NDArray):
        return Edge(p0, p1)
