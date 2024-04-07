from unittest import TestCase
import cv2
import numpy as np
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.rendering.scenerenderer import SceneRenderer
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.sceneobjects.calibrationtargets.checkerboardtarget import CheckerbordTarget
from pycvsim.core.image_utils import overlay_points_on_image
import matplotlib.pyplot as plt
import numpy.testing

scene_object = CheckerbordTarget((7, 6), (0.05, 0.05), board_thickness=0.02,
                                 color_1=(255, 255, 255), color_2=(0, 0, 0),
                                 color_bkg=(128, 0, 0), board_boundary=0.05, name="checkerboard")
cameras = [SceneCamera(pos=np.array([0.0, 0.0, -1.5]), res=(720, 720), hfov=30.0, safe_zone=100)]
renderer = SceneRenderer(cameras=cameras, objects=[scene_object])


class TestSceneCamera(TestCase):

    def test(self, plot=True, thresh=0.5):
        for _ in range(30):
            angles = np.array([np.random.uniform(low=-10, high=10, size=1)[0],
                               np.random.uniform(low=-10, high=10, size=1)[0],
                               np.random.uniform(low=-40, high=40, size=1)[0]])
            object_pos = np.random.uniform(low=-0.2, high=0.2, size=3)
            scene_object.set_pos(object_pos)
            scene_object.set_euler_angles(angles)

            lookpos = object_pos + np.random.uniform(low=-0.1, high=0.1, size=3)
            camera_pos = np.array([0.0, 0.0, -2.0]) + np.random.uniform(low=-0.5, high=0.5, size=3)
            renderer.set_camera_position(0, camera_pos)
            renderer.set_camera_lookpos(0, lookpos, np.array([0.0, 1.0, 0.0]))

            img_render = renderer.render_image(0, apply_distortion=True)
            img_gray = cv2.cvtColor(img_render, cv2.COLOR_RGB2GRAY)
            object_points = scene_object.get_object_points()
            exp_image_points = renderer.cameras[0].get_pixel_point_lies_in(object_points)
            ret, calc_image_points = cv2.findChessboardCorners(img_gray, (7, 6), None)

            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                calc_image_points = cv2.cornerSubPix(img_gray, calc_image_points, (7, 6), (-1, -1), criteria)
                img_overlayed_1 = overlay_points_on_image(img_render, exp_image_points)

                img_overlayed_2 = cv2.drawChessboardCorners(img_render, (7, 6), calc_image_points, ret)


                if plot:
                    plt.figure()
                    plt.subplot(1, 2, 1)
                    plt.imshow(img_overlayed_1)
                    plt.subplot(1, 2, 2)
                    plt.imshow(img_overlayed_2)
                    plt.show()

                calc_image_points = calc_image_points.reshape(-1, 2)
                error = np.linalg.norm(calc_image_points - exp_image_points, axis=-1)
                print(np.mean(error))
                np.testing.assert_array_less(error, thresh, "failed")
