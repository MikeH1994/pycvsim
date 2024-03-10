from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import math
import open3d as o3d
import pycvsim.core as cvmaths


class SceneCamera:
    """
    The SceneCamera class represents a virtual camera in the pbrt scene
    """
    n_cameras: int = 0
    name: str = ""

    def __init__(self, pos: NDArray = np.zeros(3), r: NDArray = np.eye(3), res: Tuple[int, int] = (640, 512),
                 hfov: float = 40.0, name: str = ""):
        """
        Creates a Camera instance using a position in space and a 3x3 rotation matrix to define the viewing direction

        :param pos: the position of the camera, in world coordinates. Shape: (3)
        :type pos: np.ndarray
        :param r: the 3x3 rotation matrix to get from the camera's coordinate space to the world coordinate space
        :type r: np.ndarray
        :param res: a tuple containing the x and y resolution of the camera
        :type res: (int, int)
        :param hfov: the horiontal field of view, in degrees
        :type hfov: float
        """
        if name == "":
            name = "camera {}".format(SceneCamera.n_cameras + 1)
        assert(pos.shape == (3, ))
        assert(r.shape == (3, 3))

        self.pos: NDArray = pos
        self.xres: int = int(res[0])
        self.yres: int = int(res[1])
        self.hfov: float = hfov
        self.vfov: float = cvmaths.hfov_to_vfov(hfov, self.xres, self.yres)
        self.r: NDArray = r
        self.name = name
        SceneCamera.n_cameras += 1

    def axes(self) -> Tuple[NDArray, NDArray, NDArray]:
        """

        :return:
        """
        return cvmaths.rotation_matrix_to_axes(self.r)

    def lookat(self, lookpos: NDArray, up: NDArray):
        """

        :param lookpos:
        :param up:
        :return:
        """
        r = cvmaths.lookpos_to_rotation_matrix(self.pos, lookpos, up)
        self.r = r

    def rotate(self, angles, degrees = True):
        """

        :param angles:
        :param degrees:
        :return:
        """
        euler_angles = cvmaths.rotation_matrix_to_euler_angles(self.r)
        if not degrees:
            angles = np.degrees(angles)
        euler_angles += angles
        r = cvmaths.euler_angles_to_rotation_matrix(euler_angles)
        self.r = r

    def translate(self, pos):
        """

        :param pos:
        :return:
        """
        self.pos += pos

    def lookpos(self) -> NDArray:
        """
        Calculates the lookpos (i.e. a point in 3D space the camera is looking at) from the camera's position and its
            3x3 rotation matrix

        :return: a point along the direction vector that the camera is looking at. Shape (3)
        :rtype: np.ndarray
        """
        return cvmaths.rotation_matrix_to_lookpos(self.pos, self.r)

    def up(self) -> NDArray:
        """
        Returns the up direction for the camera (the y axis)
        :return: a point along the direction vector that the camera is looking at. Shape (3)
        :rtype: np.ndarray
        """
        return self.axes()[1]

    def calc_pixel_direction(self, u: float, v: float) -> NDArray:
        """
        Get the direction vector corresponding to the given pixel coordinates

        :param u: the coordinates in the x direction of the image
        :type u: float
        :param v: the coordinates in the y direction of the image
        :type v: float
        :return: an array of length 3, which corresponds to the direction vector in world space for the given
                 pixel coordinates
        :rtype: np.ndarray
        """
        # define the optical centre of the
        c_x = self.xres / 2.0
        c_y = self.yres / 2.0

        # calculate the direction vector of the ray in local coordinates
        vz = 1
        vx = 2.0 * vz * (u - c_x + 0.5) / self.xres * np.tan(np.radians(self.hfov / 2.0))
        vy = 2.0 * vz * (v - c_y + 0.5) / self.yres * np.tan(np.radians(self.vfov / 2.0))
        vec = np.array([vx, vy, vz])
        # calculate the direction vector in world coordinates
        return np.matmul(self.r, vec)

    def calc_pixel_point_lies_in(self, point: NDArray) -> NDArray:
        """
        Deproject a point in 3D space on to the 2D image plane, and calculate the coordinates of it

        :param point: a point in 3D space. Shape: (3)
        :type point: np.ndarray
        :return: an array of shape (2) containing the x and y coordinates of the 3D point deprojected on to the
            image plane
        :rtype: np.ndarray
        """
        x_axis, y_axis, z_axis = self.axes()
        # calculate the direction vector from the camera to the defined point
        direction_vector = point - self.pos
        # convert this vector to local coordinate space
        x_prime = direction_vector.dot(x_axis)
        y_prime = direction_vector.dot(y_axis)
        z_prime = direction_vector.dot(z_axis)
        hfov = np.radians(self.hfov)
        vfov = np.radians(self.vfov)
        # deproject on to image plane
        k_x = 2 * z_prime * np.tan(hfov / 2.0)
        k_y = 2 * z_prime * np.tan(vfov / 2.0)
        u = ((x_prime / k_x + 0.5) * self.xres)
        v = ((y_prime / k_y + 0.5) * self.yres)
        return np.array([u, v])

    def generate_rays(self) -> NDArray:
        """
        Generate a set of rays for each pixel in Open3D's format for use in the Open3D raycasting. Each Open3D ray is
            a vector of length 6, where the first 3 elements correspond to the origin of the ray (the camera position),
            and the last 3 elements are the direction vector of the ray

        :return: a 3D array of shape (yres, xres, 6) corresponding to the open3d rays for each pixel
        :rtype: np.ndarray
        """
        return o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=self.hfov,
            center=self.lookpos(),
            eye=self.pos,
            up=self.up(),
            width_px=self.xres,
            height_px=self.yres).numpy()


    @staticmethod
    def create_camera_from_lookpos(pos: NDArray, lookpos: NDArray, up: NDArray,
                                   res: Tuple[int, int], hfov: float) -> SceneCamera:
        """
        Creates a camera from a lookpos, as opposed to a 3x3 rotation matrix.

        :param pos: the position of the camera in world coordinates. Shape (3)
        :type pos: np.ndarray
        :param lookpos: a point in world coordinates the camera is looking at. Shape (3)
        :rtype look_pos: np.ndarray
        :param up: the world direction vector corresponding to the up direction in the camera's image. Shape (3)
        :type up: np.ndarray
        :param res: a 2-tuple containing (xres, yres)- the resolution of the image in the x and y direction, in pixels
        :type res: (int, int)
        :param hfov: the horizontal field of view of the camera, in degrees
        :type hfov: float
        :return: the created camera instance
        :rtype: SceneCamera
        """
        r = cvmaths.lookpos_to_rotation_matrix(pos, lookpos, up)
        return SceneCamera(pos, r, res, hfov)

    @staticmethod
    def create_camera_from_euler_angles(pos: NDArray, euler_angles: NDArray,
                                        res: Tuple[int, int], hfov: float) -> SceneCamera:
        """
        Creates a camera from a lookpos, as opposed to a 3x3 rotation matrix.

        :param pos: the position of the camera in world coordinates. Shape (3)
        :type pos: np.ndarray
        :param euler_angles: an array of length 3, containing the euler angles, in radians
        :type euler_angles: np.ndarray
        :param res: a 2-tuple containing (xres, yres)- the resolution of the image in the x and y direction, in pixels
        :type res: (int, int)
        :param hfov: the horizontal field of view of the camera, in degrees
        :type hfov: float
        :return: the created camera instance
        :rtype: SceneCamera
        """
        r = cvmaths.euler_angles_to_rotation_matrix(euler_angles)
        return SceneCamera(pos, r, res, hfov)
