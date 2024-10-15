from numpy.typing import NDArray
from pycvsim.camera.basecamera import BaseCamera


class CameraCalibrationRoutine:
    def __init__(self):
        pass

    def run(self):
        pass

    def evaluate(self, camera: BaseCamera, camera_matrix: NDArray, distortion_coeffs: NDArray):
        true_hfov, true_vfov = camera.get_fov(include_safe_zone=False)
        # true_camera_matrix = camera.get_ca
