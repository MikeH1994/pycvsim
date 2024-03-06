from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import math

class SceneCamera:
    """
    The SceneCamera class represents a virtual camera in the pbrt scene
    """
    def __init__(self, pos: NDArray, r: NDArray, res: Tuple[int, int], hfov: float):
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
        assert(pos.shape == (3, ))
        assert(r.shape == (3, 3))

        self.pos: NDArray = pos
        self.x_res: int = int(res[0])
        self.y_res: int = int(res[1])
        self.hfov: float = hfov
        self.vfov: float = SceneCamera.hfov_to_vfov(hfov, self.x_res, self.y_res)
        self.r: NDArray = r
        self.axes: Tuple[NDArray, NDArray, NDArray] = SceneCamera.calc_camera_axes(self.r)
        self.look_pos: NDArray = SceneCamera.calc_look_pos(self.pos, self.r)

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
        c_x = self.x_res / 2.0
        c_y = self.y_res / 2.0

        # calculate the direction vector of the ray in local coordinates
        vz = 1
        vx = 2.0 * vz * (u - c_x + 0.5) / self.x_res * np.tan(np.radians(self.hfov / 2.0))
        vy = 2.0 * vz * (v - c_y + 0.5) / self.y_res * np.tan(np.radians(self.vfov / 2.0))
        vec = np.array([vx, vy, vz])
        # calculate the direction vector in world coordinates
        return np.matmul(self.r, vec)

    def generate_rays(self) -> NDArray:
        """
        Generate a set of rays for each pixel in Open3D's format for use in the Open3D raycasting. Each Open3D ray is
            a vector of length 6, where the first 3 elements correspond to the origin of the ray (the camera position),
            and the last 3 elements are the direction vector of the ray

        :return: a 3D array of shape (y_res, x_res, 6) corresponding to the open3d rays for each pixel
        :rtype: np.ndarray
        """
        rays = np.zeros((self.y_res, self.x_res, 6), dtype=np.float32)
        for y in range(self.y_res):
            for x in range(self.x_res):
                dirn = self.calc_pixel_direction(x, y)
                rays[y][x][:3] = self.pos
                rays[y][x][3:] = dirn
        return rays

    def calc_pixel_point_lies_in(self, point: NDArray) -> NDArray:
        """
        Deproject a point in 3D space on to the 2D image plane, and calculate the coordinates of it

        :param point: a point in 3D space. Shape: (3)
        :type point: np.ndarray
        :return: an array of shape (2) containing the x and y coordinates of the 3D point deprojected on to the
            image plane
        :rtype: np.ndarray
        """
        x_axis, y_axis, z_axis = self.axes
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
        u = ((x_prime / k_x + 0.5) * self.x_res)
        v = ((y_prime / k_y + 0.5) * self.y_res)
        return np.array([u, v])

    def calculate_euler_angles(self) -> NDArray:
        sy = math.sqrt(self.r[0, 0] * self.r[0, 0] + self.r[1, 0] * self.r[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(self.r[2, 1], self.r[2, 2])
            y = math.atan2(-self.r[2, 0], sy)
            z = math.atan2(self.r[1, 0], self.r[0, 0])
        else:
            x = math.atan2(-self.r[1, 2], self.r[1, 1])
            y = math.atan2(-self.r[2, 0], sy)
            z = 0
        return np.degrees(np.array([x, y, z]))

    def generate_rays(self) -> NDArray:
        """
        Generate a set of rays for each pixel in Open3D's format for use in the Open3D raycasting. Each Open3D ray is
            a vector of length 6, where the first 3 elements correspond to the origin of the ray (the camera position),
            and the last 3 elements are the direction vector of the ray

        :return: a 3D array of shape (y_res, x_res, 6) corresponding to the open3d rays for each pixel
        :rtype: np.ndarray
        """
        rays = np.zeros((self.y_res, self.x_res, 6), dtype=np.float32)
        for y in range(self.y_res):
            for x in range(self.x_res):
                dirn = self.calc_pixel_direction(x, y)
                rays[y][x][:3] = self.pos
                rays[y][x][3:] = dirn
        return rays

    @staticmethod
    def calc_camera_axes(r) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Given a 3x3 rotation matrix that defines the transformation from the camera's coordinate frame (where it square_points
            in the direction (0, 0, 1)), to the world's coordinate frame. Returns a tuple of numpy arrays,
            corresponding to the x, y and z axes in the world coordinate frame.

        :param r: the 3x3 rotation matrix
        :type r: np.ndarray
        :return: (x_axis, y_axis, z_axis) -
        :rtype: np.ndarray
        """
        # create a direction vector for each axis, then transform using the rotation matrix
        x_axis = np.matmul(r, np.array([1, 0, 0]))
        y_axis = np.matmul(r, np.array([0, 1, 0]))
        z_axis = np.matmul(r, np.array([0, 0, 1]))
        return x_axis, y_axis, z_axis

    @staticmethod
    def focal_length_to_hfov(f, xres) -> float:
        """
        Calculates the horizontal field of view, in degrees, from the focal length in the x direction
            and the x resolution. (Alternatively, passing the focal length in the y direction, with the y
            resolution provides the VFOV).

        :param f: the focal length, in pixels
        :type f: type
        :param xres: the x resolution of the camera, in pixels (i.e. the width of the image)
        :type xres: int
        :return: the hfov, in degrees
        :rtype: float
        """
        return np.degrees(2.0 * np.arctan(xres / 2.0 / f)).item()

    @staticmethod
    def hfov_to_focal_length(hfov: float, xres: int) -> float:
        """
        Calculates the x focal length, in pixels, from the hfov and the x resolution of the camera.
            (Alternatively, the y focal length can be calculated by passing the vfov and the yres)

        :param hfov: the horizontal field of view, in degrees
        :type hfov: float
        :param xres: the x resolution of the camera, in pixels (i.e. width of the image)
        :type xres: int
        :return: the focal length, in pixels
        :rtype: float
        """
        hfov = np.radians(hfov)
        return xres / 2.0 / np.tan(hfov / 2)

    @staticmethod
    def hfov_to_vfov(hfov: float, xres: int, yres: int) -> float:
        """
        Calculates the vfov, from the corresponding hfov, the xres and yres

        :param hfov: the vertical field of view, in degrees
        :type hfov: float
        :param xres: the x resolution of the camera, in pixels (i.e. the width of the image)
        :type xres: int
        :param yres: the y resolution of the camera, in pixels (i.e. the height of the image)
        :type yres: int
        :return: the vfov, in degrees
        :rtype: float
        """
        hfov_radians = np.radians(hfov)
        vfov_radians = 2 * np.arctan(np.tan(hfov_radians / 2) * float(yres) / xres)
        return np.degrees(vfov_radians).item()

    @staticmethod
    def calc_look_pos(pos: NDArray, r: NDArray) -> NDArray:
        """
        Calculates the lookpos (i.e. a point in 3D space the camera is looking at) from the camera's position and its
            3x3 rotation matrix

        :param pos: the position of the camera, in world space. Shape (3)
        :type pos: np.ndarray
        :param r: the 3x3 rotation matrix that defines the transform from the camera's local coorinate frame
            to the world coordinate frame. Shape: (3, 3)
        :return: a point along the direction vector that the camera is looking at. Shape (3)
        :rtype: np.ndarray
        """
        look_dir = np.matmul(r, np.array([0, 0, 1]))
        look_pos = pos + look_dir
        return look_pos

    @staticmethod
    def euler_angles_to_rotation_matrix(euler_angles):
        alpha, beta, gamma = euler_angles
        r_x = np.array([[1, 0, 0],
                        [0, np.cos(alpha), -np.sin(alpha)],
                        [0, np.sin(alpha), np.cos(alpha)]
                        ])

        r_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                        [0, 1, 0],
                        [-np.sin(beta), 0, np.cos(beta)]
                        ])

        r_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                        [np.sin(gamma), np.cos(gamma), 0],
                        [0, 0, 1]
                        ])
        r = np.matmul(r_z, np.matmul(r_y, r_x))
        return r

    @staticmethod
    def create_camera_from_lookpos(pos: NDArray, look_pos: NDArray, y_axis: NDArray,
                                   res: Tuple[int, int], hfov: float) -> SceneCamera:
        """
        Creates a camera from a lookpos, as opposed to a 3x3 rotation matrix.

        :param pos: the position of the camera in world coordinates. Shape (3)
        :type pos: np.ndarray
        :param look_pos: a point in world coordinates the camera is looking at. Shape (3)
        :rtype look_pos: np.ndarray
        :param y_axis: the world direction vector corresponding to the y axis ('up') in the camera's frame of reference.
            Shape (3)
        :type y_axis: np.ndarray
        :param res: a 2-tuple containing (xres, yres)- the resolution of the image in the x and y direction, in pixels
        :type res: (int, int)
        :param hfov: the horizontal field of view of the camera, in degrees
        :type hfov: float
        :return: the created camera instance
        :rtype: SceneCamera
        """
        r = np.zeros((3, 3))
        # the z axis in the camera's local coordinates defines the plane going out from the optical centre of the
        # camera to a point where the camera is looking at
        z_prime = (look_pos - pos) / np.linalg.norm(look_pos - pos)
        # the y axis is already defined in the function arguments
        y_prime = y_axis / np.linalg.norm(y_axis)
        # the x axis is then calculated as the cross product between the y and z axes
        x_prime = np.cross(y_prime, z_prime)
        x_prime /= np.linalg.norm(x_prime)
        # the rotation matrix can then be constructed by the the axes
        r[:, 0] = x_prime
        r[:, 1] = y_prime
        r[:, 2] = z_prime
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
        r = SceneCamera.euler_angles_to_rotation_matrix(np.radians(euler_angles))
        return SceneCamera(pos, r, res, hfov)
