from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from pycvsim.algorithms.esf.edge import Edge
from pycvsim.algorithms.esf.utils import normalise_data
from scipy.special import expit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from typing import Union, List


class ESF:
    esf_x: NDArray
    esf_f: NDArray
    params: NDArray
    rms: float
    angle: float
    edge: Edge

    def __init__(self, img: NDArray, edge: Edge, ** kwargs):
        self.search_region = kwargs["search_region"] if "search_region" in kwargs else 30
        self.boundary_region = kwargs["boundary_region"] if "boundary_region" in kwargs else 0
        self.distance_mode = kwargs["distance_mode"] if "distance_mode" in kwargs else "normal"
        self.img = img.astype(np.float32)
        if len(self.img.shape) == 3:
            print("Image supplied to ESF 3 channels - taking mean of channels")
            self.img = np.mean(self.img, axis=-1)
        self.edge = edge
        self.esf_x, self.esf_f = self.get_edge_profile(self.img, self.edge, search_region=self.search_region)
        self.params = self.fit(self.esf_x, self.esf_f, **kwargs)
        self.rms = self.calc_rms(self.esf_x, self.esf_f)
        self.angle = self.edge.angle

    def dfdx(self, x, epsilon: float = 1e-5):
        f1 = self.f(x)
        f2 = self.f(x + epsilon)
        return (f2 - f1) / epsilon

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        return self.fn(x, *self.params)

    def get_bounds(self, n_terms: int, bounds: List[float], additional_bounds: List[float] = None):
        if isinstance(bounds, float) or isinstance(bounds, int):
            bounds = [bounds]
        dst_bounds = []
        for i in range(n_terms):
            dst_bounds += bounds
        if additional_bounds is not None:
            dst_bounds += additional_bounds
        return dst_bounds

    def calc_rms(self, x: NDArray, f: NDArray) -> float:
        f_calc = self.f(x)
        return np.sqrt(np.mean((f_calc - f) ** 2))

    def fit(self, x: NDArray, f: NDArray, **kwargs) -> NDArray:
        raise Exception("Base function ESF.fit() called")

    # noinspection PyMethodMayBeStatic
    def get_edge_profile(self, img: NDArray, edge: Edge, search_region: int = 20, normalise=True):
        height, width = img.shape
        img = img.astype(np.float32)

        esf_x = []
        esf_f = []
        x0, y0, x1, y1 = edge.get_bounds(min_x=self.boundary_region, min_y=self.boundary_region,
                                         max_x=width-self.boundary_region-1, max_y=height-self.boundary_region-1,
                                         return_as_int=True)

        if edge.is_vertical:
            for y in range(y0, y1):
                edge_pos = edge.get_edge_x(y)
                x = np.arange(max(edge_pos-search_region, 0), min(edge_pos+search_region + 1, width-1), dtype=np.int32)
                dx = edge.distance_to_edge(x, y)
                esf_x += dx.tolist()
                esf_f += img[y, x].astype(np.float32).tolist()
        else:
            for x in range(x0, x1):
                edge_pos = edge.get_edge_y(x)
                y = np.arange(max(edge_pos-search_region, 0), min(edge_pos+search_region + 1, height-1), dtype=np.int32)
                dx = edge.distance_to_edge(x, y)
                esf_x += dx.tolist()
                k = img[y, x].tolist()
                esf_f += k

        sorted_xf = sorted(zip(esf_x, esf_f))
        esf_x = np.array([x for x, f in sorted_xf])
        esf_f = np.array([f for _, f in sorted_xf])
        if normalise:
            esf_x, esf_f = normalise_data(esf_x, esf_f)
        return esf_x, esf_f

    def plot(self, title=None, xlim=None, stride=1, new_figure=True, show=True):
        x_fit = np.linspace(np.min(self.esf_x), np.max(self.esf_x), 5000)
        y_fit = self.f(x_fit)
        if new_figure:
            plt.figure()
        if title is not None:
            plt.title(title)
        plt.scatter(self.esf_x[::stride], self.esf_f[::stride], label="data")
        plt.plot(x_fit, y_fit, label="fit")
        if xlim is not None:
            plt.xlim(xlim)
        if show:
            plt.show()

    @staticmethod
    def fn(x, *params) -> NDArray:
        raise Exception("Base function ESF.fn() called")
