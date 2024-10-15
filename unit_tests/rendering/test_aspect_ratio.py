from unittest import TestCase
import cv2
import numpy as np
from pycvsim.camera.basecamera import BaseCamera
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.targets.checkerboardtarget import CheckerbordTarget


class TestSceneCamera(TestCase):
    def test(self):
        xres, yres = (1280, 720)
        scene_object = CheckerbordTarget((2, 2), (0.05, 0.05), board_thickness=0.02,
                                         color_1=(255, 255, 255), color_2=(255, 255, 255),
                                         name="checkerboard")
        cameras = [BaseCamera(pos=np.array([0.0, 0.0, -1.0]), res=(xres, yres), hfov=20.0, safe_zone=100)]
        renderer = Panda3DRenderer(cameras=cameras, objects=[scene_object])
        img = cv2.cvtColor(renderer.render(0), cv2.COLOR_RGB2GRAY)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[img == 255] = 1

        xx, yy = np.meshgrid(np.arange(xres), np.arange(yres))
        x_min, x_max = np.min(xx[mask == 1]), np.max(xx[mask == 1])
        y_min, y_max = np.min(yy[mask == 1]), np.max(yy[mask == 1])
        width = x_max - x_min
        height = y_max - y_min
        self.assertAlmostEqual(width, height, 1)

