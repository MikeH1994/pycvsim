import numpy as np
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.camera.basecamera import BaseCamera
import matplotlib.pyplot as plt
import cv2

"""obj = CheckerbordTarget((7, 7), (0.05, 0.05), board_thickness=0.02,
                        color_2 = (255, 255, 255), color_1 = (0, 0, 0),
                        color_bkg=(128, 0, 0), board_boundary=0.05, name="checkerboard")
"""
obj = SceneObject.load_armadillo()

cameras = [
    BaseCamera.create_camera_from_lookpos(pos=np.array([0.0, 0.0, -2.0]),
                                          lookpos=np.array([0.0, 0.0, 0.0]),
                                          up=np.array([0.0, 1.0, 0.0]),
                                          res=(720, 720), hfov=60.0, safe_zone=0)
]

renderer = Panda3DRenderer(cameras=cameras, objects=[obj])


while True:
    for i in range(3):
        angles = np.random.uniform(low=-180, high=180, size=3)
        object_pos = np.random.uniform(low=-0.2, high=0.2, size=3)
        obj.set_pos(object_pos)
        obj.set_euler_angles(angles)

        lookpos = object_pos + np.random.uniform(low=-0.5, high=0.5, size=3)
        camera_pos = np.array([0.0, 0.0, -2.0]) + np.random.uniform(low=-0.5, high=0.5, size=3)
        renderer.set_camera_position(0, camera_pos)
        renderer.set_camera_lookpos(0, object_pos, np.array([0.0, 1.0, 0.0]))

        img_render = renderer.render(0, apply_distortion=True)
        img_panda = cv2.cvtColor(img_render, cv2.COLOR_RGB2GRAY)
        img_panda = (img_panda != 51).astype(np.uint8)
        img_o3d = renderer.raycast_scene(0)["object_ids"] + 1
        img_3 = np.zeros((*img_panda.shape, 3), dtype=np.uint8)
        img_3[img_panda == 1] = [255.0, 0.0, 0.0]
        img_3[img_o3d == 1] = [0.0, 255.0, 0.0]
        img_3[(img_panda == 1) & (img_o3d == 1)] = [255.0, 255.0, 0.0]

        intersection = np.sum((img_panda == 1) & (img_o3d == 1))
        union = np.sum((img_panda == 1) | (img_o3d == 1))
        iou = intersection / union

        plt.figure()
        plt.suptitle("IOU = {}".format(iou))
        plt.subplot(1, 3, 1)
        plt.imshow(img_panda)
        plt.subplot(1, 3, 2)
        plt.imshow(img_o3d)
        plt.subplot(1, 3, 3)
        plt.imshow(img_3)
        plt.show()
