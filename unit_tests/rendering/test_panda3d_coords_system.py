from unittest import TestCase
import cv2
import numpy as np
from pycvsim.camera.basecamera import BaseCamera
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.sceneobjects.sceneobject import SceneObject

scene_object = SceneObject.load_armadillo()
cameras = [BaseCamera(pos=np.array([0.0, 0.0, -2.0]), res=(720, 720), hfov=60.0, safe_zone=100)]
renderer = Panda3DRenderer(cameras=cameras, objects=[scene_object])


class TestSceneCamera(TestCase):
    def iou(self, img_panda, img_o3d, plot=True):

        intersection = np.sum((img_panda == 1) & (img_o3d == 1))
        union = np.sum((img_panda == 1) | (img_o3d == 1))
        return intersection / union

    def test_1(self, iou_threshold=0.9):
        for i in range(3):
            for _ in range(5):
                angles = np.zeros(3)
                angles[i] = np.random.uniform(low=-80, high=80, size=1)

                scene_object.set_euler_angles(angles)
                img_render = renderer.render(0, apply_distortion=True)
                img_1 = (cv2.cvtColor(img_render, cv2.COLOR_RGB2GRAY) != 51).astype(np.uint8)
                img_2 = renderer.raycast_scene(0)["object_ids"] + 1
                assert(self.iou(img_1, img_2) > iou_threshold)

    def test_2(self, iou_threshold=0.9):
        for i in range(3):
            for _ in range(5):
                angles = np.random.uniform(low=-50, high=50, size=3)
                scene_object.set_euler_angles(angles)
                img_render = renderer.render(0, apply_distortion=True)
                img_1 = (cv2.cvtColor(img_render, cv2.COLOR_RGB2GRAY) != 51).astype(np.uint8)
                img_2 = renderer.raycast_scene(0)["object_ids"] + 1
                assert(self.iou(img_1, img_2) > iou_threshold)

    def test_3(self, iou_threshold=0.9):
        for i in range(3):
            for _ in range(5):
                angles = np.random.uniform(low=-50, high=50, size=3)
                object_pos = np.random.uniform(low=-0.2, high=0.2, size=3)
                scene_object.set_pos(object_pos)
                scene_object.set_euler_angles(angles)

                lookpos = object_pos + np.random.uniform(low=-0.5, high=0.5, size=3)
                camera_pos = np.array([0.0, 0.0, -2.0]) + np.random.uniform(low=-0.5, high=0.5, size=3)
                renderer.set_camera_position(0, camera_pos)
                renderer.set_camera_lookpos(0, lookpos, np.array([0.0, 1.0, 0.0]))

                img_render = renderer.render(0, apply_distortion=True)
                img_1 = (cv2.cvtColor(img_render, cv2.COLOR_RGB2GRAY) != 51).astype(np.uint8)
                img_2 = renderer.raycast_scene(0)["object_ids"] + 1
                assert(self.iou(img_1, img_2) > iou_threshold)
