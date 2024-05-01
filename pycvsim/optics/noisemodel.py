from typing import Tuple
from numpy.typing import NDArray
import numpy as np
from pycvsim.core.utils import rescale_image


class NoiseModel:
    def __init__(self, image_size: Tuple[int, int], preset: str = "none", **kwargs):
        self.image_size: Tuple[int, int] = image_size
        self.p_hot_pixels: float = 0.0  # probability of a given pixel being hot
        self.p_dead_pixels: float = 0.0  # probability of a given pixel being dead
        self.p_blinking_pixels: float = 0.0  # probability of given pixel blinking
        self.p_columnisation: float = 0.0  # probability of a column displaying column noise
        self.hot_pixel_value: float = 255.0
        self.dead_pixel_value: float = 0.0
        self.gain_sigma: float = 0.1
        self.offset_sigma: float = 5.0
        self.temporal_noise_sigma: float = 15.0
        self.columnisation_gain_mean: float = 1.2
        self.columnisation_gain_sigma: float = 0.2
        self.columnisation_offset_mean: float = 0.0
        self.columnisation_offset_sigma: float = 10.0

        empty_indices = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        self.dead_pixels: Tuple[NDArray, NDArray] = empty_indices
        self.hot_pixels: Tuple[NDArray, NDArray] = empty_indices
        self.blinking_pixels: Tuple[NDArray, NDArray] = empty_indices
        self.fpn_columns: Tuple[NDArray, NDArray] = empty_indices

        self.gain_map: NDArray = np.ones(self.image_size, dtype=np.float32)
        self.offset_map: NDArray = np.zeros(self.image_size, dtype=np.float32)

        self.process_arguments(preset, **kwargs)
        self.generate_pixel_maps()
        self.generate_pixel_maps()

    def process_arguments(self, preset: str, **kwargs):
        if preset == "default":
            self.p_hot_pixels: float = 0.001
            self.p_dead_pixels: float = 0.001
            self.p_blinking_pixels: float = 0.001
            self.p_columnisation: float = 0.0

        kwargs_map = self.get_kwargs_map()
        for kwarg, value in kwargs.items():
            assert(kwarg in kwargs_map)
            member_variable, expected_type = kwargs_map[kwarg]
            value = expected_type(value)
            self.__dict__[member_variable] = value

    def get_kwargs_map(self):
        return {
            "p_hot": ("p_hot_pixels", float),
            "p_dead": ("p_dead_pixels", float),
            "p_blinking": ("p_blinking_pixles", float),
            "p_columnisation": ("p_columnisation", float),
            "hot_pixel_value": ("hot_pixel_value", float),
            "dead_pixel_value": ("dead_pixel_value", float),
            "gain_sigma": ("gain_sigma", float),
            "offset_sigma": ("offset_sigma", float),
            "temporal_noise": ("temporal_noise_sigma", float),
            "columnisation_gain_mean": ("columnisation_gain_mean", float),
            "columnisation_gain_sigma": ("columnisation_gain_sigma", float),
            "columnisation_offset_mean": ("columnisation_offset_mean", float),
            "columnisation_offset_sigma": ("columnisation_offset_sigma", float)
        }

    def generate_pixel_maps(self):
        dead_map = np.random.uniform(size=self.image_size) < self.p_dead_pixels
        hot_map = np.random.uniform(size=self.image_size) < self.p_hot_pixels
        blinking_map = np.random.uniform(size=self.image_size) < self.p_blinking_pixels
        fpn_colmns = np.random.uniform(size=self.image_size[0]) < self.p_columnisation
        fpn_columns_map = np.zeros(self.image_size, dtype=np.uint8)
        fpn_columns_map[fpn_colmns] = 1
        self.dead_pixels = np.where(dead_map)
        self.hot_pixels = np.where(hot_map & ~dead_map)
        self.blinking_pixels = np.where(blinking_map & ~dead_map & ~hot_map)
        self.fpn_columns = np.where(fpn_columns_map)
        self.gain_map = np.random.normal(loc=1.0, scale=self.gain_sigma, size=self.image_size)
        self.offset_map = np.random.normal(scale=self.offset_sigma, size=self.image_size)

    def apply(self, image: NDArray, dst_dtype=np.uint8) -> NDArray:
        reshaped_size = image.shape if len(image.shape) == 2 else (*image.shape[:2], 1)
        image = np.copy(image).astype(np.float32)
        image *= self.gain_map.reshape(reshaped_size)
        image += self.offset_map.reshape(reshaped_size)
        temporal_noise = np.random.normal(scale=self.temporal_noise_sigma, size=image.shape[:2]).reshape(reshaped_size)
        image += temporal_noise
        image[self.dead_pixels] = self.dead_pixel_value
        image[self.hot_pixels] = self.hot_pixel_value

        blinking_p_shape = self.blinking_pixels[0].shape if len(image.shape) == 2 else (*self.blinking_pixels[0].shape, 1)
        blinking_p = np.random.uniform(size=blinking_p_shape)
        blinking_values = np.full(shape=blinking_p_shape, fill_value=self.dead_pixel_value)
        blinking_values[blinking_p > 0.5] = self.hot_pixel_value
        image[self.blinking_pixels] = blinking_values

        return rescale_image(image, np.uint8, kind="clip")
