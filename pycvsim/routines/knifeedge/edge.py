from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2
from typing import Union, List, Tuple


class Edge:
    def __init__(self, image: NDArray, p0: NDArray, p1: NDArray):
        if p0[0] < p1[0]:
            self.x0, self.y0 = p0
            self.x1, self.y1 = p1
        else:
            self.x1, self.y1 = p0
            self.x0, self.y0 = p1
        self.min_x, self.max_x = min(self.x0, self.x1), max(self.x0, self.x1)
        self.min_y, self.max_y = min(self.y0, self.y1), max(self.y0, self.y1)
        self.image = image
        self.m = (self.y1 - self.y0) / (self.x1 - self.x0)
        self.c = self.y0 - self.x0*self.m
        self.angle = self.gradient_to_angle(self.m)
        self.is_vertical = self.gradient_is_vertical(self.m)

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

        return (y - self.m * x - self.c) / np.sqrt(self.m ** 2 + 1)

    # noinspection PyMethodMayBeStatic
    def gradient_to_angle(self, m: float):
        return np.degrees(np.arctan(m))

    # noinspection PyMethodMayBeStatic
    def gradient_is_vertical(self, m: float):
        return np.abs(m) > 1

    # noinspection PyMethodMayBeStatic
    def normalise_data(self, esf_x: NDArray, esf_f: NDArray):
        def fn(x, x0_, y0_, a_, b_):
            f = y0_ + a_ * expit((x - x0_) / b_)
            return f

        # if the left side of the plot is greater than the right side, invert it
        if np.mean(esf_f[esf_x < 0]) > np.mean(esf_f[esf_x > 0]):
            esf_x *= -1.0
        p0 = [0, np.min(esf_f), np.max(esf_f) - np.min(esf_f), 1]
        [x0, y0, a, _], _ = curve_fit(fn, esf_x, esf_f, p0=p0)

        lower_val = y0
        upper_val = y0 + a
        x_offset = x0

        esf_f -= lower_val
        esf_f /= (upper_val - lower_val)
        esf_x -= x_offset

        return esf_x, esf_f

    # noinspection PyMethodMayBeStatic
    def get_edge_profile(self, boundary: int = 10, search_region: int = 10, normalise=True):
        height, width = self.image.shape[:2]
        image = self.image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.astype(np.float32)

        esf_x = []
        esf_f = []
        if self.is_vertical:
            y0, y1 = int(self.min_y+boundary), int(self.max_y-boundary)
            for y in range(y0, y1+1):
                x_edge = self.get_edge_x(y)
                x0, x1 = int(max(x_edge - search_region, 0)), int(min(x_edge + search_region, width-1))
                for x in range(x0, x1+1):
                    dx = self.distance_to_edge(x, y)
                    esf_x.append(dx)
                    esf_f.append(float(image[y][x]))
        else:
            x0, x1 = int(self.min_x+boundary), int(self.max_x-boundary)
            for x in range(x0, x1+1):
                y_edge = self.get_edge_y(x)
                y0, y1 = int(max(y_edge - search_region, 0)), int(min(y_edge + search_region, height-1))
                for y in range(y0, y1+1):
                    dx = self.distance_to_edge(x, y)
                    esf_x.append(dx)
                    esf_f.append(float(image[y][x]))

        sorted_xf = sorted(zip(esf_x, esf_f))
        esf_x = np.array([x for x, f in sorted_xf])
        esf_f = np.array([f for _, f in sorted_xf])

        if normalise:
            esf_x, esf_f = self.normalise_data(esf_x, esf_f)
        return esf_x, esf_f

    def get_edge_x(self, y):
        # y = mx + c, therefore, x = (y - c)/m
        return (y - self.c) / self.m

    def get_edge_y(self, x):
        return self.m * x + self.c