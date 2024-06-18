from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import InterpolatedUnivariateSpline
from typing import Union, List, Tuple
import skimage.filters.thresholding


def clamp(a, min_val, max_val):
    if a < min_val:
        return int(min_val)
    if a > max_val:
        return int(max_val)
    return int(a)


class Edge:
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

    def get_bounds(self, width, height, return_as_int = False):
        x0, y0, x1, y1 = self.x0, self.y0, self.x1, self.y1
        if self.is_vertical:
            assert(y0 <= y1)
            if y0 < 0.0:
                y0 = 0.0
                x0 = self.get_edge_x(self.y0)
            if y1 > height - 1.0:
                y1 = height - 1.0
                x1 = self.get_edge_x(self.y1)
        else:
            assert(x0 <= x1)
            if x0 < 0.0:
                x0 = 0.0
                y0 = self.get_edge_y(x0)
            if x1 > width - 1.0:
                x1 = width - 1.0
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
    def gradient_is_vertical(self, m: float):
        return np.abs(m) > 1

    def get_edge_x(self, y):
        # y = mx + c, therefore, x = (y - c)/m
        return (y - self.c) / self.m if not self.inf_gradient else self.x0

    def get_edge_y(self, x):
        return self.m * x + self.c

    @staticmethod
    def create_from_image(image: NDArray, threshold: float = None, orientation: str = "auto"):
        def line_eqn(x, m, c):
            return m * x + c
        assert(orientation == "auto" or orientation == "horizontal" or orientation == "vertical")
        assert(len(image.shape) == 2)

        image = image.astype(np.float32)

        if threshold is None:
            threshold = skimage.filters.thresholding.threshold_otsu(image)

        if orientation == "auto":
            edge_x_v, edge_y_v = Edge.get_edge_points(image, "vertical", threshold)
            edge_x_h, edge_y_h = Edge.get_edge_points(image, "horizontal", threshold)
            if edge_x_v.shape[0] > edge_x_h.shape[0]:
                edge_x = edge_x_v
                edge_y = edge_y_v
            else:
                edge_x = edge_x_h
                edge_y = edge_y_h
        else:
            edge_x, edge_y = Edge.get_edge_points(image, orientation, threshold)

        assert(edge_x.shape[0] > 2)
        params, _ = curve_fit(line_eqn, edge_x, edge_y)
        m, c = params[0], params[1]
        x0 = np.min(edge_x)
        y0 = m*x0 + c
        x1 = np.max(edge_x)
        y1 = m*x1 + c

        return Edge(np.array([x0, y0]), np.array([x1, y1]))

    @staticmethod
    def get_edge_points(image, orientation, threshold):
        assert(orientation == "horizontal" or orientation == "vertical")
        assert(len(image.shape) == 2)
        assert(image.dtype == np.float32 or image.dtype == np.float64)
        edge_x = []
        edge_y = []
        height, width = image.shape
        if orientation == "horizontal":
            for x in range(width):
                y = Edge.get_edge_position(image[:, x], threshold)
                if y is not None:
                    edge_x.append(x)
                    edge_y.append(y)
        else:
            for y in range(height):
                x = Edge.get_edge_position(image[y], threshold)
                if x is not None:
                    edge_x.append(x)
                    edge_y.append(y)
        edge_x = np.array(edge_x)
        edge_y = np.array(edge_y)
        return edge_x, edge_y

    @staticmethod
    def get_edge_position(data, threshold, display: bool = False):
        assert(len(data.shape) == 1)
        data = data.astype(np.float32)
        data_offset = data - threshold
        indices = np.arange(data.shape[0])
        spline = InterpolatedUnivariateSpline(indices, data_offset)
        roots = spline.roots()

        if display:
            plt.plot(indices, data)
            plt.axhline(y=threshold)
            for r in roots:
                plt.axvline(x=r)
            plt.show()
        if len(roots) == 1:
            return roots[0]
        else:
            return None
