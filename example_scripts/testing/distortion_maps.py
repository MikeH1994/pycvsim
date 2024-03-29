import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycvsim.rendering.distortionmodel import DistortionModel
import time

w = 2
safe_zone = 50
image_size = (640, 520)

image_safe_zone = np.full((image_size[1] + 2*safe_zone, image_size[0] + 2*safe_zone, 3), fill_value=255, dtype=np.uint8)
for x in range(0, image_safe_zone.shape[1], 40):
    image_safe_zone[:, x - w: x + w + 1, :] = 0
for y in range(0, image_safe_zone.shape[0], 40):
    image_safe_zone[y - w:y + w + 1, :, :] = 0

image = image_safe_zone[safe_zone:-safe_zone, safe_zone:-safe_zone, :]

# Canera matrix
camera_matrix = np.array([[1710.0, 0.0, 320.0],
                          [0.0, 1710.0, 256.0],
                          [0.0, 0.0, 1.0]])
k = 2.0
distortion_coeffs = np.array([k*-0.8424,  k*0.1724, k*-0.00101, k*-0.006596, k*4.3341])

distortion_model = DistortionModel(camera_matrix, distortion_coeffs, image_size)
distortion_model_safe_zone = DistortionModel(camera_matrix, distortion_coeffs, image_size, safe_zone=safe_zone)
image_distorted = distortion_model.distort_image(image)
image_safe_zone_distorted = distortion_model_safe_zone.distort_image(image_safe_zone)
image_safe_zone_distorted_cropped = image_safe_zone_distorted[safe_zone:-safe_zone, safe_zone:-safe_zone, :]


plt.imshow(image_distorted)
plt.figure()
plt.imshow(image_safe_zone_distorted_cropped)
plt.figure()
plt.imshow(image_distorted - image_safe_zone_distorted_cropped)
plt.show()

image_undistorted = distortion_model.undistort_image(image_distorted)
image_undistorted_safe_zone = distortion_model_safe_zone.undistort_image(image_safe_zone_distorted)
image_undistorted_safe_zone_cropped = image_undistorted_safe_zone[safe_zone:-safe_zone, safe_zone:-safe_zone, :]

plt.imshow(image_distorted)
plt.figure()
plt.imshow(image_undistorted)
plt.figure()
plt.imshow(image_undistorted_safe_zone_cropped)
plt.figure()
plt.imshow(image_undistorted - image_undistorted_safe_zone_cropped)
plt.show()
