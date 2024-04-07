from pycvsim.datageneration.calibrationmanager import CalibrationManager
from pycvsim.rendering import SceneRenderer, SceneCamera
from pycvsim.sceneobjects.calibrationtargets import CheckerbordTarget
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pycvsim.core.image_utils import overlay_points_on_image
from calibration.device import Device
from calibration.stereo import StereoPair
from pycvsim.core.pinhole_camera_maths import fov_to_focal_length


def run(board_size=(7, 6), resolution=(800, 800)):
    cameras = [SceneCamera.create_camera_from_lookpos(pos=np.array([-0.2, 0.0, -1.5]), lookpos=np.array([0.0, 0.0, 0.0]),
                                                      up=np.array([0.0, 1.0, 0.0]), res=resolution, hfov=30.0),
               SceneCamera.create_camera_from_lookpos(pos=np.array([0.2, 0.0, -1.5]), lookpos=np.array([0.0, 0.0, 0.0]),
                                                      up=np.array([0.0, 1.0, 0.0]), res=resolution, hfov=30.0)]
    devices = [Device("Camera {}".format(i+1), resolution) for i in range(len(cameras))] # , Device("Right Camera", resolution)
    calibration_target = CheckerbordTarget(board_size, (0.05, 0.05), board_thickness=0.02, color_bkg=(128, 0, 0))
    manager = CalibrationManager(cameras=cameras, calibration_target=calibration_target, n_horizontal=9, n_vertical=9)
    setpoints = manager.generate_setpoints()
    images = manager.run(setpoints)
    for key_index, image_setpoint in enumerate(images):
        for i in range(len(devices)):
            device = devices[i]
            image = image_setpoint[i]
            success, image_overlayed = device.add_calibration_point(image, key_index, board_size)
            plt.imshow(image_overlayed)
            plt.show()

    object_points = calibration_target.get_object_points(transformed=False)
    for i, device in enumerate(devices):
        print("rms = {:.3f}".format(device.calibrate(object_points)))
        print("Camera matrix:")
        print(cameras[i].camera_matrix)
        print("Calculated camera matrix:")
        print(device.camera_matrix)
        fovx, fovy, f, principalPoint, _ = cv2.calibrationMatrixValues(device.camera_matrix, device.image_size,
                                                                       device.image_size[0], device.image_size[1])
        print("Fox: {}, {}".format(fovx, fovy))
        print("Focal length: {}".format(f))
        print("Focal length (2): {}".format(fov_to_focal_length(fovx, device.image_size[0])))
    #stereo_pair = StereoPair(devices[0], devices[1], object_points)
    #print(stereo_pair.T_1)
    #print(stereo_pair.T_2)


if __name__ == "__main__":
    run()
