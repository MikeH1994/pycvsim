from unittest import TestCase
import cv2
import pycv
import numpy as np
from pycvsim.rendering import Renderer
from pycvsim.targets import CheckerboardTarget
from pycv import PinholeCamera
import matplotlib.pyplot as plt

board_size = (7, 6)
scene_object = CheckerboardTarget(board_size, (0.05, 0.05), 0.01, (0.08, 0.08))
xres = 640
yres = 512
f = pycv.fov_to_focal_length(30.0, xres)
camera_matrix = pycv.create_camera_matrix(f, f, xres/2, yres/2)
camera = PinholeCamera(camera_matrix, (xres, yres), p=np.array([0.0, 0.0, -0.5]))
renderer = Renderer(cameras=[camera], objects=[scene_object])

lookpos = np.mean(scene_object.get_object_points(), axis=0)
camera_pos = np.array([0.0, 0.0, -1.0])
renderer.cameras[0].position = camera_pos
renderer.cameras[0].set_lookpos(lookpos, np.array([0.0, 1.0, 0.0]))

img_render = renderer.render(0, n_samples=128, return_as_8_bit=False)[:, :, 0].astype(np.float32)
img_gray = pycv.convert_to_8_bit(img_render, return_as_rgb=True)
object_points = scene_object.get_object_points()
corners_expected = renderer.cameras[0].project_points_to_2d(object_points, return_distorted=False)
success, corners_found, img_overlayed = pycv.find_checkerboard_corners(img_render, board_size, winSize=(11,11))

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
