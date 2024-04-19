import unittest
from unittest import TestCase
import cv2
import numpy as np
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.sceneobjects.targets.checkerboardtarget import CheckerbordTarget
from pycvsim.core.image_utils import overlay_points_on_image
import matplotlib.pyplot as plt

board_size = (7, 6)
scene_object = CheckerbordTarget(board_size, (0.05, 0.05), board_thickness=0.02,
                                 color_1=(255, 255, 255), color_2=(0, 0, 0),
                                 color_bkg=(128, 0, 0), board_boundary=0.05, name="checkerboard")
renderer = Panda3DRenderer(objects=[scene_object])


class TestDistortionCoeffs(TestCase):

    def test_distort_points(self, thresh=0.001):
        """
        (1) check that distort_map matches distort_points
        :return:
        """
        for distortion_coeffs in [np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                                  np.array([-3.2, 0.32, -0.01, -0.02, 4.3])]:
            camera = SceneCamera(pos=np.array([0.0, 0.0, -2.0]), res=(720, 640), hfov=15.0,
                                 distortion_coeffs=distortion_coeffs, safe_zone=200)
            mdl = camera.distortion_model

            # check that opencv map and spline match
            height, width = mdl.distort_map_x.shape
            x, y = np.arange(width), np.arange(height)
            xx, yy = np.meshgrid(x, y)
            x1 = mdl.distort_map_x[yy, xx]
            y1 = mdl.distort_map_y[yy, xx]
            p2 = mdl.distort_points(np.stack([xx, yy], axis=-1), remove_safe_zone=False)
            x2, y2 = p2[:, :, 0], p2[:, :, 1]
            error_1 = np.abs(x2 - x1)
            error_2 = np.abs(y2 - y1)
            np.testing.assert_array_less(error_1, thresh)
            np.testing.assert_array_less(error_2, thresh)

            x1 = mdl.distort_map_x[yy, xx]
            x2 = mdl.distort_map_x_fn(yy, xx, grid=False)
            y1 = mdl.distort_map_y[yy, xx]
            y2 = mdl.distort_map_y_fn(yy, xx, grid=False)
            error_1 = np.abs(x2 - x1)
            error_2 = np.abs(y2 - y1)
            np.testing.assert_array_less(error_1, thresh)
            np.testing.assert_array_less(error_2, thresh)

    @unittest.skip
    def test_distort_checkerboards(self, plot=False):
        image_size = (1280, 1280)
        hfov = 15.0
        distortion_coeffs = np.array([-3.2, 0.32, -0.01, -0.02, 4.3])
        safe_zone = 200
        camera = SceneCamera(pos=np.array([0.0, 0.0, -2.0]), res=image_size, hfov=hfov,
                             distortion_coeffs=distortion_coeffs, safe_zone=safe_zone)
        renderer.remove_all_cameras()
        renderer.add_camera(camera)
        img = renderer.render_image(0, apply_distortion=False)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_distorted = renderer.render_image(0)
        img_distorted_gray = cv2.cvtColor(img_distorted, cv2.COLOR_RGB2GRAY)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        _, points = cv2.findChessboardCorners(img_gray, board_size, None)
        points = cv2.cornerSubPix(img_gray, points, board_size, (-1, -1), criteria)

        _, points_distorted = cv2.findChessboardCorners(img_distorted_gray, board_size, None)
        points_distorted = cv2.cornerSubPix(img_distorted_gray, points_distorted, board_size, (-1, -1), criteria)
        points_undistorted = camera.distortion_model.undistort_points(points_distorted)

        error = np.abs(points - points_undistorted)
        print(error)

        img = cv2.drawChessboardCorners(img, board_size, points, True)
        img = overlay_points_on_image(img, points_undistorted.reshape(-1, 2))
        img_distorted = cv2.drawChessboardCorners(img_distorted, board_size, points_distorted, True)

        if plot:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(img_distorted)
            plt.show()

    @unittest.skip
    def test_distort_image(self, thresh=0.1):
        """

        :return:
        """
        for image_size in [(640, 512), (1200, 720)]:
            for hfov in [20.0, 40.0, 60.0]:
                for safe_zone in [100, 200, 250]:
                    for distortion_coeffs in [np.array([-0.8424, 0.1724, -0.00101, -0.006596, 4.3341])]:
                        xres, yres = image_size
                        camera = SceneCamera(pos=np.array([0.0, 0.0, -1.5]), res=image_size, hfov=hfov,
                                             distortion_coeffs=distortion_coeffs, safe_zone=safe_zone)
                        renderer.remove_all_cameras()
                        renderer.add_camera(camera)
                        img = renderer.render_image(0, apply_distortion=False)
                        img_distorted = renderer.render_image(0, remove_safe_zone=False)
                        img_undistorted = camera.distortion_model.undistort_image(img_distorted)
                        error = np.abs(img_undistorted - img)
                        error = np.mean(cv2.erode(error, np.ones((7, 7), np.uint8), iterations=1), axis=-1)
                        np.testing.assert_array_less(error, thresh, "failed")
