from pycvsim.sceneobjects.calibrationtargets import CalibrationTarget
import numbers


class CalibrationManager:
    def __init__(self, calibration_target: CalibrationTarget, **kwargs):
        self.calibration_target: CalibrationTarget = calibration_target
        self.camera_mode: str = "fixed"
        self.max_rotation: float = 45.0
        self.distance: float = 3.0
        self.n_images: int = 15

        if "camera_mode" in kwargs:
            assert(isinstance(self.camera_mode, str))
            self.camera_mode = kwargs["camera_mode"]
        if "max_rotation" in kwargs:
            assert(isinstance(self.max_rotation, numbers.Number))
            self.camera_mode = kwargs["max_rotation"]

    def run(self):
        if self.camera_mode == "fixed":
            for n in range(self.n_images):
                pass
        else:
            pass