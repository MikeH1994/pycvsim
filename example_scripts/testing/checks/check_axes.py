import numpy as np
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.core.globalsettings import GlobalSettings
import cv2
import itertools


def check_intersection(img_panda, img_o3d, iou_threshold=0.9):
    intersection = np.sum((img_panda == 1) & (img_o3d == 1))
    union = np.sum((img_panda == 1) | (img_o3d == 1))
    return intersection / union > iou_threshold


def test_1(renderer, scene_object):
    n_passed = 0
    n_total = 0
    for i in range(3):
        for _ in range(5):
            angles = np.zeros(3)
            angles[i] = np.random.uniform(low=-80, high=80, size=1)

            scene_object.set_euler_angles(angles)
            img_render = renderer.render(0, apply_distortion=True)
            img_1 = (cv2.cvtColor(img_render, cv2.COLOR_RGB2GRAY)!= 51).astype(np.uint8)
            img_2 = renderer.raycast_scene(0)["object_ids"] + 1

            n_total += 1
            if check_intersection(img_1, img_2):
                n_passed += 1
    return n_passed/n_total*100.0


def test_2(renderer, scene_object):
    n_passed = 0
    n_total = 0
    for i in range(3):
        for _ in range(5):
            angles = np.random.uniform(low=-50, high=50, size=3)
            scene_object.set_euler_angles(angles)
            img_render = renderer.render(0, apply_distortion=True)
            img_1 = (cv2.cvtColor(img_render, cv2.COLOR_RGB2GRAY) != 51).astype(np.uint8)
            img_2 = renderer.raycast_scene(0)["object_ids"] + 1

            n_total += 1
            if check_intersection(img_1, img_2):
                n_passed += 1
    return n_passed / n_total * 100.0


def test_3(renderer, scene_object):
    n_passed = 0
    n_total = 0
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

            n_total += 1
            if check_intersection(img_1, img_2):
                n_passed += 1
    return n_passed / n_total * 100.0


def run():
    obj = SceneObject.load_armadillo()
    cameras = [SceneCamera(pos=np.array([0.0, 0.0, -2.0]), res=(720, 720), hfov=60.0)]
    renderer = Panda3DRenderer(cameras=cameras, objects=[obj])
    for perm_1 in list(itertools.permutations(['x', 'y', 'z'])):
        GlobalSettings.OPEN3D_EULER_AXES = perm_1[0] + perm_1[1] + perm_1[2]
        for perm_2 in list(itertools.permutations(['x', 'y', 'z'])):
            GlobalSettings.PANDA3D_EULER_AXES = perm_2[0] + perm_2[1] + perm_2[2]

            print("Using axis Open3d {} and Panda3d {}".format(GlobalSettings.OPEN3D_EULER_AXES,  GlobalSettings.PANDA3D_EULER_AXES))
            print("    test 1 success: {:.2f}%".format(test_1(renderer, obj)))
            print("    test 2 success: {:.2f}%".format(test_2(renderer, obj)))
            print("    test 3 success: {:.2f}%".format(test_3(renderer, obj)))


if __name__ == "__main__":
    run()
