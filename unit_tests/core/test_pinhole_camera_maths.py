from unittest import TestCase
import cv2
import pycvsim.core.pinhole_camera_maths as cvmaths
import numpy as np
from numpy.testing import assert_allclose

class TestPinholeCameraMaths(TestCase):
    def test_focal_length_to_hfov(self):
        for (fx, fy) in [(1, 1), (15, 12), (142, 87)]:
            for image_size in [(32, 32), (640, 512), (640, 720), (199, 162)]:
                width, height = image_size
                cx, cy = (width-1)/2, (height-1)/2
                camera_matrix = np.array([[fx, 0.0, cx],
                                          [0.0, fy, cy],
                                          [0.0, 0.0, 1.0]], dtype=np.float32)
                exp_fovx, exp_fovy, _, _, _ = cv2.calibrationMatrixValues(camera_matrix, image_size, 1, 1)
                calc_fovx = cvmaths.focal_length_to_fov(fx, width)
                calc_fovy = cvmaths.focal_length_to_fov(fy, height)
                assert_allclose(exp_fovx, calc_fovx, rtol=1e-4)
                assert_allclose(exp_fovy, calc_fovy, rtol=1e-4)

                calc_fx = cvmaths.fov_to_focal_length(calc_fovx, width)
                assert_allclose(fx, calc_fx, rtol=1e-4)
                calc_fy = cvmaths.fov_to_focal_length(calc_fovy, height)
                assert_allclose(fy, calc_fy, rtol=1e-4)


    def test_hfov_to_focal_length(self):
        self.fail()

    def hfov_to_vfov(self):
        self.fail()
