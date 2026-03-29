from unittest import TestCase

from unittest import TestCase
import cv2
import pycv
import numpy as np
from pycvsim.rendering import Renderer
from pycvsim.targets import CheckerboardTarget
from pycv import PinholeCamera
import matplotlib.pyplot as plt

board_size = (7, 6)
scene_object = CheckerboardTarget(board_size, (0.05, 0.05), 0.01, (0.05, 0.05))
xres = 640
yres = 512
f = pycv.fov_to_focal_length(30.0, xres)
camera_matrix = pycv.create_camera_matrix(f, f, xres/2, yres/2)
camera = PinholeCamera(camera_matrix, (xres, yres), p=np.array([0.0, 0.0, -0.5]))
renderer = Renderer(cameras=[camera], objects=[scene_object])

multisamples = renderer.get_multisample_pattern(64, np.array([0, 1]), np.array([0, 0]))

plt.scatter(multisamples[0, :, 0].flatten(), multisamples[0, :, 1].flatten())
plt.scatter(multisamples[1, :, 0].flatten(), multisamples[1, :, 1].flatten())
plt.show()