from pycvsim.datageneration.calibrationmanager import CalibrationManager
from pycvsim.rendering import SceneRenderer, SceneCamera
from pycvsim.sceneobjects.calibrationtargets import CheckerbordTarget
import numpy as np
import matplotlib.pyplot as plt
from pycvsim.core.image_utils import overlay_points_on_image


def run():
    cameras = [SceneCamera(pos=np.array([0.0, 0.0, -1.5]), res=(1280, 1024), hfov=30.0)]
    calibration_target = CheckerbordTarget((6, 5), (0.05, 0.05), board_thickness=0.02, color_bkg=(128, 0, 0))
    manager = CalibrationManager(cameras=cameras, calibration_target=calibration_target)
    setpoints = manager.generate_setpoints()
    images = manager.run(setpoints)
    for image_setpoint in images:
        for image in image_setpoint:
            plt.figure()
            plt.imshow(image)
            plt.show()


if __name__ == "__main__":
    run()
