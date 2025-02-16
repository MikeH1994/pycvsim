import numpy as np
import cv2
import scipy.interpolate
from typing import Tuple
from numpy.typing import NDArray
from scipy.optimize import minimize


class DistortionModel:
    interpolation_method = cv2.INTER_LINEAR

    def __init__(self, camera_matrix: NDArray, distortion_coeffs: NDArray, image_size: Tuple[int, int],
                 safe_zone: int = 0):
        assert(distortion_coeffs.shape == (5,))
        self.safe_zone = safe_zone
        self.width, self.height = image_size
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.undistort_map_x, self.undistort_map_y = None, None
        self.undistort_map_x_fn, self.undistort_map_y_fn = None, None
        self.distort_map_x, self.distort_map_y = None, None
        self.distort_map_x_fn, self.distort_map_y_fn = None, None
        self.initialise(safe_zone=safe_zone)

    def initialise(self, safe_zone=0):
        self.safe_zone = safe_zone
        image_size = (self.width + 2*safe_zone, self.height + 2*safe_zone)
        camera_matrix = np.copy(self.camera_matrix)
        camera_matrix[0][2] += safe_zone
        camera_matrix[1][2] += safe_zone
        distortion_coeffs = self.distortion_coeffs
        x, y = np.arange(image_size[0]), np.arange(image_size[1])

        self.undistort_map_x, self.undistort_map_y = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeffs, None,
                                                                                 None, image_size, cv2.CV_32FC1)
        self.undistort_map_x_fn = scipy.interpolate.RectBivariateSpline(y, x, self.undistort_map_x)
        self.undistort_map_y_fn = scipy.interpolate.RectBivariateSpline(y, x, self.undistort_map_y)

        self.distort_map_x, self.distort_map_y = self.invert_maps(self.undistort_map_x, self.undistort_map_y)
        self.distort_map_x_fn = scipy.interpolate.RectBivariateSpline(y, x, self.distort_map_x)
        self.distort_map_y_fn = scipy.interpolate.RectBivariateSpline(y, x, self.distort_map_y)

    def distort_image(self, image: NDArray, remove_safe_zone=True):
        """

        :param image:
        :return:
        """
        assert(image.shape[:2] == self.distort_map_x.shape)
        img = cv2.remap(image, self.distort_map_x, self.distort_map_y, self.interpolation_method)
        if remove_safe_zone:
            img = img[self.safe_zone: -self.safe_zone, self.safe_zone: -self.safe_zone, :]
        return img

    def undistort_image(self, image: NDArray, remove_safe_zone=True):
        """

        :param image:
        :return:
        """
        assert(image.shape[:2] == self.undistort_map_x.shape)
        img = cv2.remap(image, self.undistort_map_x, self.undistort_map_y, self.interpolation_method)
        if remove_safe_zone:
            img = img[self.safe_zone: -self.safe_zone, self.safe_zone: -self.safe_zone, :]
        return img

    def distort_points(self, points: NDArray, remove_safe_zone=True):
        """

        :param points:
        :param remove_safe_zone:
        :return:
        """
        assert(points.shape[-1] == 2)

        if remove_safe_zone:
            points += self.safe_zone

        init_shape = points.shape
        points = points.reshape(-1, 2)
        dst_points = np.zeros(points.shape, dtype=np.float32)
        dst_points[:, 0] = self.distort_map_x_fn(points[:, 1], points[:, 0], grid=False)
        dst_points[:, 1] = self.distort_map_y_fn(points[:, 1], points[:, 0], grid=False)
        if remove_safe_zone:
            points -= self.safe_zone
        return dst_points.reshape(init_shape)

    def undistort_points(self, points: NDArray, remove_safe_zone=True):
        """

        :param points:
        :param remove_safe_zone:
        :return:
        """
        if remove_safe_zone:
            points += self.safe_zone

        assert(points.shape[-1] == 2)
        init_shape = points.shape
        points = points.reshape(-1, 2)
        dst_points = np.zeros(points.shape, dtype=np.float32)
        dst_points[:, 0] = self.undistort_map_x_fn(points[:, 1], points[:, 0], grid=False)
        dst_points[:, 1] = self.undistort_map_y_fn(points[:, 1], points[:, 0], grid=False)
        if remove_safe_zone:
            points -= self.safe_zone
        return dst_points.reshape(init_shape)

    def invert_maps(self, map_x, map_y):
        F = np.zeros((map_x.shape[0], map_x.shape[1], 2), dtype=np.float32)
        # unsure why you need to add 0.5 here, but it works
        F[:, :, 0] = map_x
        F[:, :, 1] = map_y

        (h, w) = F.shape[:2]  # (h, w, 2), "xymap"
        I = np.zeros_like(F)
        I[:, :, 1], I[:, :, 0] = np.indices((h, w))  # identity map
        P = np.copy(I)
        k = 0.5
        for i in range(30):
            correction = I - cv2.remap(F, P, None, interpolation=self.interpolation_method)
            P += correction * k
            k *= 0.5
        return P[:, :, 0], P[:, :, 1]

    @staticmethod
    def remap_points(x, y, camera_matrix, distortion_coeffs):
        """

        :param x:
        :param y:
        :param camera_matrix:
        :param distortion_coeffs:
        :return:
        """
        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]
        k1, k2, p1, p2, k3 = distortion_coeffs

        x = (x - cx) / fx
        y = (y - cy) / fy
        r = np.sqrt(x**2 + y**2)

        x_dist = x * (1 + k1*r**2 + k2*r**4 + k3*r**6) + (2 * p1 * x * y + p2 * (r**2 + 2 * x * x))
        y_dist = y * (1 + k1*r**2 + k2*r**4 + k3*r**6) + (p1 * (r**2 + 2 * y * y) + 2 * p2 * x * y)
        x_dist = x_dist * fx + cx
        y_dist = y_dist * fy + cy
        return x_dist.astype(np.float32), y_dist.astype(np.float32)