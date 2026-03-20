from pycvsim.core.kwargs_parser import KwargsParser
from pycvsim.camera.virtualcamera import VirtualCamera

class Camera(VirtualCamera):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def create_rgb_camera(**kwargs):
        parser = KwargsParser(**kwargs)
