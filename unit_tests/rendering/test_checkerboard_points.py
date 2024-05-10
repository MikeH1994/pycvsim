from unittest import TestCase
import cv2
import numpy as np
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.open3drenderer import Open3DRenderer
from pycvsim.sceneobjects.targets.checkerboardtarget import CheckerbordTarget
from pycvsim.core.image_utils import overlay_points_on_image
import matplotlib.pyplot as plt

board_size = (7, 6)
scene_object = CheckerbordTarget(board_size, (0.05, 0.05), board_thickness=0.02,
                                 color_1=(255, 255, 255), color_2=(0, 0, 0),
                                 color_bkg=(128, 0, 0), board_boundary=0.05, name="checkerboard")
cameras = [SceneCamera(pos=np.array([0.0, 0.0, -1.5]), res=(720, 720), hfov=30.0, safe_zone=100),
           SceneCamera(pos=np.array([0.0, 0.0, -1.5]), res=(605, 599), hfov=40.0, safe_zone=50)]
panda3d_renderer = Panda3DRenderer(cameras=cameras, objects=[scene_object])
open3d_renderer = Open3DRenderer(cameras=cameras, objects=[scene_object])



class TestSceneCamera(TestCase):
    def test(self, plot=False, thresh=0.2):
        for _ in range(30):
            angles = np.array([np.random.uniform(low=-10, high=10, size=1)[0],
                               np.random.uniform(low=-10, high=10, size=1)[0],
                               np.random.uniform(low=-40, high=40, size=1)[0]])
            object_pos = np.random.uniform(low=-0.2, high=0.2, size=3)
            scene_object.set_pos(object_pos)
            scene_object.set_euler_angles(angles)

            lookpos = object_pos + np.random.uniform(low=-0.1, high=0.1, size=3)
            camera_pos = np.array([0.0, 0.0, -2.0]) + np.random.uniform(low=-0.5, high=0.5, size=3)
            for renderer in [open3d_renderer, panda3d_renderer]:
                for camera_index in range(len(cameras)):
                    for return_as_8_bit in [True, False]:
                        renderer.set_camera_position(camera_index, camera_pos)
                        renderer.set_camera_lookpos(camera_index, lookpos, np.array([0.0, 1.0, 0.0]))

                        img_render = renderer.render_image(0, n_samples=16, return_as_8_bit=return_as_8_bit).astype(np.uint8)
                        img_gray = cv2.cvtColor(img_render, cv2.COLOR_RGB2GRAY)
                        object_points = scene_object.get_object_points()[::-1]
                        exp_image_points = renderer.cameras[0].get_pixel_point_lies_in(object_points)
                        ret, calc_image_points = cv2.findChessboardCorners(img_gray, board_size, None)

                        if not ret:
                            raise Exception("Setup error")

                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        calc_image_points = cv2.cornerSubPix(img_gray, calc_image_points, (11, 11), (-1, -1), criteria)
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
                        test_name = "({}): Renderer: {}, cam index: {} 8 bit: " \
                                    "{}\n     angles: ({})".format(_, type(renderer).__name__, camera_index,
                                                               return_as_8_bit, angles)
                        print("{}: {}".format(test_name, np.mean(error)))
                        self.assertLess(np.mean(error), thresh, "Test failed: {}".format(test_name))
