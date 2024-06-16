from unittest import TestCase
import cv2
import numpy as np
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.open3drenderer import Open3DRenderer
from pycvsim.targets.checkerboardtarget import CheckerbordTarget
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
    def test(self, plot=False, thresh=0.5):
        for _ in range(30):
            for camera_index in range(len(cameras)):
                for return_as_8_bit in [True, False]:
                    camera = cameras[camera_index]
                    angles = np.array([np.random.uniform(low=-10, high=10, size=1)[0],
                                       np.random.uniform(low=-10, high=10, size=1)[0],
                                       np.random.uniform(low=-40, high=40, size=1)[0]])
                    object_pos = np.random.uniform(low=-0.2, high=0.2, size=3)
                    scene_object.set_pos(object_pos)
                    scene_object.set_euler_angles(angles)

                    lookpos = object_pos + np.random.uniform(low=-0.1, high=0.1, size=3)
                    camera_pos = np.array([0.0, 0.0, -2.0]) + np.random.uniform(low=-0.5, high=0.5, size=3)
                    images = []
                    for renderer in [open3d_renderer, panda3d_renderer]:
                        renderer.set_camera_position(camera_index, camera_pos)
                        renderer.set_camera_lookpos(camera_index, lookpos, np.array([0.0, 1.0, 0.0]))
                        img_render = renderer.render(camera_index, n_samples=4, return_as_8_bit=return_as_8_bit).astype(np.uint8)
                        img_gray = cv2.cvtColor(img_render, cv2.COLOR_RGB2GRAY)
                        images.append(img_gray)

                    img_1 = images[0]
                    img_2 = images[1]
                    err = img_1 - img_2
                    n = np.count_nonzero(np.abs(err) > 0.0)
                    iou = 1.0 - n / (camera.xres * camera.yres)

                    if plot:
                        plt.imshow(err)
                        plt.show()

                    test_name = "({}): cam index: {} 8 bit: {}\n     angles: ({})".format(_, camera_index,
                                                                                          return_as_8_bit, angles)
                    print("{}".format(test_name))
                    print("    {}".format(iou))

                    self.assertGreater(iou, 0.99, "failed")
