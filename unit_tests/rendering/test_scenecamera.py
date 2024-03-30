from unittest import TestCase
import numpy as np
from numpy.testing import assert_allclose
from pycvsim.rendering.scenecamera import SceneCamera

class TestSceneCamera(TestCase):
    def test_axes(self):
        self.fail()

    def test_calc_pixel_direction(self):
        self.fail()

    def test_calc_pixel_point_lies_in(self):
        self.fail()

    def test_generate_rays(self):
        cameras = [
            SceneCamera.create_camera_from_euler_angles(pos=np.array([0.2, 0.2, -1.5]),
                                                        euler_angles=np.array([0, 0, 0]),
                                                        res=(640, 512), hfov=30.0),
            SceneCamera.create_camera_from_lookpos(pos=np.array([0.8, 0.8, -1.5]),
                                                   lookpos=np.array([0.0, 0.0, 0.0]),
                                                   up=np.array([0.0, 1.0, 0.0]),
                                                   res=(320, 256), hfov=100.0),
            SceneCamera.create_camera_from_euler_angles(pos=np.array([0.8, 0.8, 0.8]),
                                                        euler_angles=np.random.uniform(-180, 180, 3),
                                                        res=(320, 256), hfov=100.0)
        ]
        for camera in cameras:
            expected_rays = np.zeros((camera.yres, camera.xres, 6), dtype=np.float32)
            for y in range(camera.yres):
                for x in range(camera.xres):
                    dirn = camera.get_pixel_direction(x, y)
                    expected_rays[y][x][:3] = camera.pos
                    expected_rays[y][x][3:] = dirn
            calculated_rays = camera.generate_rays()
            assert_allclose(expected_rays, calculated_rays, atol=1e-6)

    def test_lookpos(self):
        # create a camera from
        self.fail()

    def test_create_camera_from_lookpos(self):
        self.fail()

    def test_create_camera_from_euler_angles(self):
        self.fail()
