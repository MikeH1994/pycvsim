from pycvsim.core.kwargs_parser import KwargsParser
from pycvsim.camera.basecamera import BaseCamera

class Camera(BaseCamera):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def create_rgb_camera(**kwargs):
        parser = KwargsParser(**kwargs)
