import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycvsim.rendering.distortionmodel import DistortionModel
import time


image_size = (640, 512)
image = np.full((512, 640, 3), fill_value=255,  dtype=np.uint8)
w = 2
for x in range(40, image_size[0], 40):
    image[:, x-w: x+w+1, :] = 0
for y in range(40, image_size[1], 40):
    image[y-w:y+w+1, :, :] = 0


# Canera matrix
camera_matrix = np.array([[1710.0, 0.0, 320.0],
                          [0.0, 1710.0, 256.0],
                          [0.0, 0.0, 1.0]])
distortion_coeffs = np.array([-0.8424,  0.1724, -0.00101, -0.006596, 4.3341])

t1 = time.time()
distortion_model = DistortionModel(camera_matrix, distortion_coeffs, image_size)
t2 = time.time()
print("Took {:.3f}s".format(t2-t1))

image_distorted = distortion_model.distort_image(image)
image_undistorted = distortion_model.undistort_image(image_distorted)

plt.imshow(image)
plt.figure()
plt.imshow(image_distorted)
plt.figure()
plt.imshow(image_undistorted)
plt.show()