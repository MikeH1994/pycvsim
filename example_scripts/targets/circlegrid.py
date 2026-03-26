from unittest import TestCase
import cv2
import pycv
import numpy as np
from pycvsim.rendering.open3drenderer import Open3DRenderer
from pycvsim.targets import CircleGridTarget
from pycv import PinholeCamera
import matplotlib.pyplot as plt

board_size = (7, 6)
scene_object = CircleGridTarget(board_size, 0.05, 0.01, (0, 0, 0),
                                (255, 255, 255))
xres = 640
yres = 512
f = pycv.fov_to_focal_length(30.0, xres)
camera_matrix = pycv.create_camera_matrix(f, f, xres/2, yres/2)
camera = PinholeCamera(camera_matrix, (xres, yres), p=np.array([0.0, 0.0, -0.5]))
renderer = Open3DRenderer(cameras=[camera], objects=[scene_object])

lookpos = np.mean(scene_object.get_object_points(), axis=0)
camera_pos = np.array([0.0, 0.0, -1.0])
renderer.set_camera_position(0, camera_pos)
renderer.set_camera_lookpos(0, lookpos, np.array([0.0, 1.0, 0.0]))

img_render = renderer.render(0, n_samples=64, return_as_8_bit=True)
img_gray = cv2.cvtColor(img_render, cv2.COLOR_RGB2GRAY)
object_points = scene_object.get_object_points()
corners_expected = renderer.cameras[0].project_points_to_2d(object_points, return_distorted=False)
success, corners_found, img_overlayed = pycv.find_circles_grid(img_gray, board_size)

plt.figure()
plt.subplot(121)
plt.imshow(img_overlayed)
plt.subplot(122)
plt.imshow(img_gray, cmap='gray')
plt.scatter(corners_expected[:, 0], corners_expected[:, 1], c='r')
plt.show()

corners_found = corners_found.reshape(-1, 2)
error = np.linalg.norm(corners_found - corners_expected, axis=-1)
print("res: {}".format(np.mean(error)))
