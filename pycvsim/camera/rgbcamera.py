from pycvsim.camera.basecamera import BaseCamera
import numpy as np
from numpy.typing import NDArray

class RGBCamera(BaseCamera):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_image_to_rgb(self, image: NDArray):
        assert(image.shape[-1] == 3)
        return image