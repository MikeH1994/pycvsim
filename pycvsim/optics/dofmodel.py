import numpy as np
import cv2
import scipy.interpolate
from typing import Tuple
from numpy.typing import NDArray
from scipy.optimize import minimize
import scipy.ndimage


class DOFModel:
    def __init__(self, focal_length: float, f_number: float):
        self.focal_length = focal_length
        self.f_number = f_number

    def get_kernel(self, focus_distance, distance: float) -> NDArray:
        return np.ones((5, 5))

    def apply(self, image: NDArray, depth_map: NDArray, focus_distance: float, n_steps=100):
        """

        :param image: the
        :param depth_map:
        :param focus_distance:
        :param n_steps:
        :return:
        """
        dtype = image.dtype
        image = image.astype(np.float32)
        distances = np.percentile(depth_map.reshape(-1), np.linspace(0.0, 100.0, n_steps))
        image_stack = np.zeros((*image.shape, n_steps))
        for i, distance in enumerate(distances):
            kernel = self.get_kernel(focus_distance, distance)
            image_stack[:, :, i] = scipy.ndimage.convolvlf.gee(image, kernel)
        interp_fn, indices = self.generate_interpolation_fn(image_stack, distances)
        return interp_fn(indices, distances, grid=False).astype(dtype)

    def generate_interpolation_fn(self, image_stack: NDArray, distances: NDArray):
        height, width, stack_length = image_stack.shape
        assert(stack_length == distances.shape[0])
        image_stack = image_stack.reshape((-1, stack_length))
        indices = np.arange(stack_length)

        distances__, indices__ = np.meshgrid(distances, indices)
        interp_fn = scipy.interpolate.RectBivariateSpline(indices__, distances__, image_stack)

        return interp_fn, indices.reshape((height, width))
