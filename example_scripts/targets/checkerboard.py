from unittest import TestCase
import cv2
import pycv
import numpy as np
from pycvsim.rendering.open3drenderer import Open3DRenderer
from pycvsim.targets import CheckerboardTarget
from pycv import PinholeCamera
import matplotlib.pyplot as plt

board_size = (7, 6)
scene_object = CheckerboardTarget(board_size, (0.05, 0.05), board_thickness=0.02,
                                  color_1=(255, 255, 255), color_2=(0, 0, 0),
                                  color_bkg=(128, 0, 0), board_boundary=0.05, name="checkerboard")

xres = 640
yres = 512
f = pycv.fov_to_focal_length(30.0, xres)
camera_matrix = pycv.create_camera_matrix(f, f, xres/2, yres/2)
camera = PinholeCamera(camera_matrix, (xres, yres), p=np.array([0.0, 0.0, -0.5]))
renderer = Open3DRenderer(cameras=[camera], objects=[scene_object])


angles = np.array([np.random.uniform(low=-10, high=10, size=1)[0],
                   np.random.uniform(low=-10, high=10, size=1)[0],
                   np.random.uniform(low=-40, high=40, size=1)[0]])
r = pycv.euler_angles_to_rotation_matrix(angles)
object_pos = np.random.uniform(low=-0.2, high=0.2, size=3)
scene_object.set_pos(object_pos)
scene_object.set_rotation(r)

lookpos = object_pos + np.random.uniform(low=-0.1, high=0.1, size=3)
camera_pos = np.array([0.0, 0.0, -1.0]) + np.random.uniform(low=-0.5, high=0.5, size=3)
renderer.set_camera_position(0, camera_pos)
renderer.set_camera_lookpos(0, lookpos, np.array([0.0, 1.0, 0.0]))

img_render = renderer.render(0, n_samples=512, return_as_8_bit=True)
img_gray = cv2.cvtColor(img_render, cv2.COLOR_RGB2GRAY)
object_points = scene_object.get_object_points()[::-1]
corners_expected = renderer.cameras[0].project_points_to_2d(object_points, return_distorted=False)
success, corners_found, img_overlayed = pycv.find_checkerboard_corners(img_gray, (board_size[0]-1, board_size[1]-1))

plt.figure()
plt.subplot(121)
plt.imshow(img_overlayed)
plt.subplot(122)
plt.imshow(img_gray)
plt.scatter(corners_expected[:, 0], corners_expected[:, 1], c='r')
plt.show()

corners_found = corners_found.reshape(-1, 2)
error = np.linalg.norm(corners_found - corners_expected, axis=-1)
print("res: {}".format(np.mean(error)))
