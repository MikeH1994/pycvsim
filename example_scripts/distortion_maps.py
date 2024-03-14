import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycvsim.rendering.distortionmodel import DistortionModel


def compute_distortion_map(camera_matrix, distortion_coeffs, size):
    fx = camera_matrix[0][0]
    fy = camera_matrix[1][1]
    cx = camera_matrix[0][2]
    cy = camera_matrix[1][2]
    k1, k2, p1, p2, k3 = distortion_coeffs
    width, height = size

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - cx) / fx
    y = (y - cy) / fy
    r = np.sqrt(x ** 2 + y ** 2)

    x_dist = x * (1 + k1*r**2 + k2*r**4 + k3*r**6) + (2*p1*x*y + p2 * (r**2 + 2*x**2))
    y_dist = y * (1 + k1*r**2 + k2*r**4 + k3*r**6) + (p1 * (r**2 + 2*y**2) + 2 * p2*x*y)
    x_dist = x_dist * fx + cx
    y_dist = y_dist * fy + cy

    return x_dist.astype(np.float32), y_dist.astype(np.float32)


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


distortion_x, distortion_y = compute_distortion_map(camera_matrix, distortion_coeffs, image_size)
undistort_map_x, undistort_map_y = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeffs, None,
                                                               None, image_size, cv2.CV_32FC1)
print(np.max(np.abs(distortion_x - undistort_map_x)))
print(np.max(np.abs(distortion_y - undistort_map_y)))


image_distorted = cv2.remap(image, distortion_x, distortion_y, cv2.INTER_LINEAR)
image_undistorted = cv2.undistort(image, camera_matrix, distortion_coeffs, None, None)

plt.imshow(image)
plt.figure()
plt.imshow(image_distorted)
plt.figure()
plt.imshow(image_undistorted)
plt.figure()
plt.imshow(image_undistorted_distorted)
plt.show()