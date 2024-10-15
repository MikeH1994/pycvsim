import unittest
from unittest import TestCase
import numpy as np
from numpy.testing import assert_allclose
from pycvsim.camera.basecamera import BaseCamera
import open3d as o3d


def compute_pixel_dirn_old(camera, u, v):
    cx = camera.cx
    cy = camera.cy

    # calculate the direction vector of the ray in local coordinates
    vz = 1
    vx = 2.0 * vz * (u - cx + 0.5) / camera.xres * np.tan(np.radians(camera.hfov / 2.0))
    vy = 2.0 * vz * (v - cy + 0.5) / camera.yres * np.tan(np.radians(camera.vfov / 2.0))
    vec = np.array([vx, vy, vz])
    # calculate the direction vector in world coordinates
    vec = np.matmul(camera.r, vec)
    vec /= np.linalg.norm(vec)
    return vec


class TestSceneCamera(TestCase):
    def test_axes(self):
        self.fail()

    @unittest.SkipTest
    def test_calc_pixel_direction(self):
        cameras = [BaseCamera(res=(720, 640), hfov=35.0),
                   BaseCamera.create_camera_from_euler_angles(pos=np.array([0.2, 0.2, -1.5]),
                                                              euler_angles=np.array([0, 0, 0]),
                                                              res=(640, 512), hfov=30.0),
                   BaseCamera.create_camera_from_lookpos(pos=np.array([0.8, 0.8, -1.5]),
                                                         lookpos=np.array([0.0, 0.0, 0.0]),
                                                         up=np.array([0.0, 1.0, 0.0]),
                                                         res=(320, 256), hfov=100.0),
                   BaseCamera.create_camera_from_euler_angles(pos=np.array([0.8, 0.8, 0.8]),
                                                              euler_angles=np.random.uniform(-180, 180, 3),
                                                              res=(320, 256), hfov=100.0)
                   ]
        np.random.seed(12345)
        for camera in cameras:
            # check 1d array passed
            for x in range(camera.xres):
                for y in range(camera.yres):
                    expected = compute_pixel_dirn_old(camera, x, y)
                    calculated = camera.get_pixel_direction(np.array([x, y]))
                    assert_allclose(expected, calculated, atol=1e-6)

            # check 2d array passed
            p = np.random.randint(low=0, high=min(camera.xres-1, camera.yres-1), size=(100, 2))
            calculated_direction = camera.get_pixel_direction(p)
            for i in range(p.shape[0]):
                expected_direction = compute_pixel_dirn_old(camera, p[i, 0], p[i, 1])
                assert_allclose(expected_direction, calculated_direction[i], atol=1e-6)

            # check 3d array passed
            p = np.random.randint(low=0, high=min(camera.xres-1, camera.yres-1), size=(20, 20, 2))
            calculated_direction = camera.get_pixel_direction(p)
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    expected_direction = compute_pixel_dirn_old(camera, p[i, j, 0], p[i, j, 1])
                    assert_allclose(expected_direction, calculated_direction[i, j], atol=1e-6)

            # check 4d array passed
            p = np.random.randint(low=0, high=min(camera.xres-1, camera.yres-1), size=(20, 20, 20, 2))
            calculated_direction = camera.get_pixel_direction(p)
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    for k in range(p.shape[2]):
                        expected_direction = compute_pixel_dirn_old(camera, p[i, j, k, 0], p[i, j, k, 1])
                        assert_allclose(expected_direction, calculated_direction[i, j, k], atol=1e-6)

    def test_calc_pixel_point_lies_in(self):
        self.fail()

    def test_generate_rays(self):
        cameras = [
            BaseCamera(res=(640, 512), hfov=30.0, optical_center=((640-1)/2, (512-1)/2)),
            BaseCamera.create_camera_from_euler_angles(pos=np.array([0.2, 0.2, -1.5]),
                                                       euler_angles=np.array([0, 0, 0]),
                                                       res=(320, 256), hfov=30.0),
            BaseCamera.create_camera_from_lookpos(pos=np.array([0.8, 0.8, -1.5]),
                                                  lookpos=np.array([0.0, 0.0, 0.0]),
                                                  up=np.array([0.0, 1.0, 0.0]),
                                                  res=(320, 256), hfov=100.0),
            BaseCamera.create_camera_from_euler_angles(pos=np.array([0.8, 0.8, 0.8]),
                                                       euler_angles=np.random.uniform(-180, 180, 3),
                                                       res=(320, 256), hfov=100.0)
        ]
        for camera_index, camera in enumerate(cameras):
            expected_rays = np.zeros((camera.yres, camera.xres, 6), dtype=np.float32)
            for y in range(camera.yres):
                for x in range(camera.xres):
                    dirn = camera.get_pixel_direction(np.array([x, y]))
                    expected_rays[y][x][:3] = camera.pos
                    expected_rays[y][x][3:] = dirn
            calculated_rays_o3d = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                fov_deg=camera.hfov, center=camera.get_lookpos(), eye=camera.pos,
                up=camera.get_up(), width_px=camera.xres, height_px=camera.yres).numpy()

            calculated_rays = camera.generate_rays().reshape(camera.yres, camera.xres, 6)
            assert_allclose(expected_rays, calculated_rays_o3d, atol=1e-6, err_msg="{} failed (o3d)".format(camera_index))
            assert_allclose(expected_rays, calculated_rays, atol=1e-6, err_msg="{} failed".format(camera_index))

    def test_lookpos(self):
        # create a camera from
        self.fail()

    def test_get_pixel_direction_and_get_pixel_point_lies_in_match(self):
        camera = BaseCamera.create_camera_from_euler_angles(pos=np.array([0.8, 0.8, 0.8]),
                                                            euler_angles=np.random.uniform(-180, 180, 3),
                                                            res=(320, 320), hfov=100.0)

        pixels_arr = np.random.uniform(high=camera.xres, size=(100, 2))
        for i in range(pixels_arr.shape[0]):
            pixels = pixels_arr[i]
            pixel_direction = camera.get_pixel_direction(pixels)
            pixel_object_points = camera.pos + pixel_direction*1.5
            pixels_calculated = camera.get_pixel_point_lies_in(pixel_object_points)
            assert_allclose(pixels_calculated, pixels, atol=1e-6)

    def test_create_camera_from_lookpos(self):
        self.fail()

    def test_create_camera_from_euler_angles(self):
        self.fail()
