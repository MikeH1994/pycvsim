from __future__ import annotations
from typing import Tuple, Union
import numpy as np
from numpy.typing import NDArray
import pycvsim.core as cvmaths
import scipy.spatial.transform
from pycvsim.optics.distortionmodel import DistortionModel


class SceneCamera:
    """
    The SceneCamera class represents a virtual camera in the pbrt scene
    """
    n_cameras: int = 0
    name: str = ""

    def __init__(self, pos: NDArray = np.zeros(3), r: NDArray = np.eye(3), res: Tuple[int, int] = (640, 512),
                 hfov: float = 40.0, name: str = "", optical_center: Tuple[float, float] = None,
                 distortion_coeffs: NDArray = np.zeros(5), safe_zone: int = 100):
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
        self.image_size = (self.xres, self.yres)
        self.cx, self.cy = optical_center if optical_center is not None else ((self.xres-1)/2, (self.yres-1)/2)
        self.hfov: float = hfov
        self.vfov: float = cvmaths.hfov_to_vfov(hfov, self.xres, self.yres)
        self.r: NDArray = r
        self.name: str = name
        self.safe_zone: int = safe_zone
        self.distortion_coeffs: NDArray = distortion_coeffs
        self.camera_matrix = self.get_camera_matrix()
        self.distortion_model: DistortionModel = DistortionModel(self.camera_matrix, self.distortion_coeffs,
                                                                 self.image_size, safe_zone=safe_zone)
        self.saved_state = {}
        self.save_state()
        SceneCamera.n_cameras += 1

    def get_camera_matrix(self):
        hfov, vfov = self.get_fov(include_safe_zone=False)
        fx = cvmaths.fov_to_focal_length(hfov, self.xres)
        fy = cvmaths.fov_to_focal_length(vfov, self.yres)
        return cvmaths.create_camera_matrix(self.cx, self.cy, fx, fy)

    def get_focal_length(self):
        fx = cvmaths.fov_to_focal_length(self.hfov, self.xres)
        fy = cvmaths.fov_to_focal_length(self.vfov, self.yres)
        return fx, fy

    def get_fov(self, include_safe_zone=True) -> Tuple[float, float]:
        hfov = self.hfov
        vfov = cvmaths.hfov_to_vfov(hfov, self.xres, self.yres)
        if include_safe_zone:
            xres_safe_zone, yres_safe_zone = self.get_res(include_safe_zone=True)
            hfov = cvmaths.calculate_hfov_for_safe_zone(self.xres, xres_safe_zone, hfov)
            vfov = cvmaths.hfov_to_vfov(hfov, xres_safe_zone, yres_safe_zone)
        return hfov, vfov

    def get_res(self, include_safe_zone=True):
        xres = self.xres
        yres = self.yres
        if include_safe_zone:
            xres += 2*self.safe_zone
            yres += 2*self.safe_zone
        return xres, yres

    def set_safe_zone(self, safe_zone):
        self.safe_zone = safe_zone
        if self.distortion_model is not None:
            self.distortion_model.initialise(safe_zone=safe_zone)

    def reset_state(self):
        self.pos = self.saved_state["object_pos"]
        self.r = self.saved_state["r"]

    def save_state(self):
        self.saved_state = {
            "object_pos": np.copy(self.pos),
            "r": np.copy(self.r)
        }

    def set_lookpos(self, lookpos: NDArray, up: NDArray):
        """

        :param lookpos:
        :param up:
        :return:
        """
        r = cvmaths.lookpos_to_rotation_matrix(self.pos, lookpos, up)
        self.r = r

    def set_euler_angles(self, angles, degrees=True, mode='absolute'):
        assert(mode == 'absolute' or mode == 'relative')
        if not degrees:
            angles = np.degrees(angles)

        if mode == 'absolute':
            r = cvmaths.euler_angles_to_rotation_matrix(angles, degrees=degrees)
        else:
            euler_angles = cvmaths.rotation_matrix_to_euler_angles(self.r, degrees=degrees)
            euler_angles += angles
            r = cvmaths.euler_angles_to_rotation_matrix(euler_angles, degrees=degrees)
        self.r = r

    def set_pos(self, pos: NDArray, mode = 'absolute'):
        assert(mode == 'absolute' or mode == 'relative')
        if mode == 'absolute':
            self.pos = pos
        else:
            self.pos += pos

    def get_axes(self) -> Tuple[NDArray, NDArray, NDArray]:
        """

        :return:
        """
        return cvmaths.rotation_matrix_to_axes(self.r)

    def get_lookpos(self) -> NDArray:
        """
        Calculates the lookpos (i.e. a points in 3D space the camera is looking at) from the camera's position and its
            3x3 rotation matrix

        :return: a points along the direction vector that the camera is looking at. Shape (3)
        :rtype: np.ndarray
        """
        return cvmaths.rotation_matrix_to_lookpos(self.pos, self.r)

    def get_up(self) -> NDArray:
        """
        Returns the up direction for the camera (the y axis)
        :return: a points along the direction vector that the camera is looking at. Shape (3)
        :rtype: np.ndarray
        """
        return self.get_axes()[1]

    def get_3d_point_from_pixel(self, u: float, v: float, distance: float) -> NDArray:
        return self.pos + distance*self.get_pixel_direction(np.array([u, v]))

    def get_pixel_direction(self, p: Union[NDArray], apply_distortion=False) -> NDArray:
        """
        Get the direction vector corresponding to the given pixel coordinates

        :param p: the pixel coordinates
        :type u: float
        :return: an array of length 3, which corresponds to the direction vector in world space for the given
                 pixel coordinates
        :rtype: np.ndarray
        """
        init_shape = p.shape
        p = p.reshape(-1, 2)
        n = p.shape[0]

        if apply_distortion:
            pass  # p = self.distortion_model.distort_points(p)

        u = p[:, 0]
        v = p[:, 1]

        # calculate the direction vector of the rays in local coordinates
        vz = 1

        vec = np.zeros((n, 3))
        vec[:, 0] = 2.0 * vz * (u - self.cx) / self.xres * np.tan(np.radians(self.hfov / 2.0))
        vec[:, 1] = 2.0 * vz * (v - self.cy) / self.yres * np.tan(np.radians(self.vfov / 2.0))
        vec[:, 2] = vz

        # calculate the direction vector in world coordinates
        r = scipy.spatial.transform.Rotation.from_matrix(self.r)
        vec = r.apply(vec)
        vec /= np.linalg.norm(vec, axis=-1).reshape(-1, 1)

        # reshape to match input size
        vec = vec.reshape((*init_shape[:-1], 3))
        return vec

    def get_pixel_point_lies_in(self, points: NDArray, apply_distortion=False) -> NDArray:
        """
        Deproject a point in 3D space on to the 2D image_safe_zone plane, and calculate the coordinates of it

        :param points: a point in 3D space. Shape: (3)
        :type points: np.ndarray
        :return: an array of shape (2) containing the x and y coordinates of the 3D point deprojected on to the
            image_safe_zone plane
        :rtype: np.ndarray
        """
        init_shape = points.shape
        points = points.reshape(-1, 3)
        x_axis, y_axis, z_axis = self.get_axes()
        # calculate the direction vector from the camera to the defined points
        direction_vector = (points - self.pos)
        # convert this vector to local coordinate space by doing dot product of
        # direction vector and each axis
        x_prime = np.sum(direction_vector*x_axis, axis=-1)
        y_prime = np.sum(direction_vector*y_axis, axis=-1)
        z_prime = np.sum(direction_vector*z_axis, axis=-1)
        hfov = np.radians(self.hfov)
        vfov = np.radians(self.vfov)
        # deproject on to image plane
        k_x = 2 * z_prime * np.tan(hfov / 2.0)
        k_y = 2 * z_prime * np.tan(vfov / 2.0)
        u = (x_prime / k_x * self.xres + self.cx)
        v = (y_prime / k_y * self.yres + self.cy)
        #
        result = np.zeros((x_prime.shape[0], 2))
        result[:, 0] = u
        result[:, 1] = v
        # returned shape is the same as initial shape, but final dimension is 2 instead of 3
        result = result.reshape((*init_shape[:-1], 2))

        if apply_distortion:
            pass # result = self.distortion_model.distort_points(result)

        return result

    def generate_rays(self, apply_distortion: bool = True, pixel_coords: NDArray = None) -> NDArray:
        """
        Generate a set of rays for each pixel in Open3D's format for use in the Open3D raycasting. Each Open3D ray is
            a vector of length 6, where the first 3 elements correspond to the origin of the ray (the camera position),
            and the last 3 elements are the direction vector of the ray

        :param apply_distortion:
        :type apply_distortion: bool
        :param pixel_coords: the (x, y) pixels to sample at. Have any number of dimensions, as long as the last dimension is 2.
        :type pixel_coords: np.ndarray
        :return: a 3D array of shape (yres, xres, n_samples, 6) corresponding to the open3d rays for each pixel
        :rtype: np.ndarray
        """

        if pixel_coords is None:
            xx, yy = np.meshgrid(np.arange(self.xres), np.arange(self.yres))
            pixel_coords = np.zeros((*xx.shape, 2), dtype=np.float32)
            pixel_coords[:, :, 0] = xx
            pixel_coords[:, :, 1] = yy

        init_shape = pixel_coords.shape
        pixel_coords = pixel_coords.reshape((-1, 2))
        pixel_direction = self.get_pixel_direction(pixel_coords, apply_distortion=apply_distortion)
        rays = np.zeros((pixel_coords.shape[0], 6), dtype=np.float32)
        rays[:, :3] = self.pos
        rays[:, 3:] = pixel_direction
        rays = rays.reshape((*init_shape[:-1], 6))
        return rays

    @staticmethod
    def create_camera_from_lookpos(pos: NDArray, lookpos: NDArray, up: NDArray,
                                   res: Tuple[int, int], hfov: float, name="",
                                   optical_center: Tuple[float, float] = None,
                                   distortion_coeffs: NDArray = np.zeros(5), safe_zone: int = 100) -> SceneCamera:
        """
        Creates a camera from a lookpos, as opposed to a 3x3 rotation matrix.

        :param pos: the position of the camera in world coordinates. Shape (3)
        :type pos: np.ndarray
        :param lookpos: a points in world coordinates the camera is looking at. Shape (3)
        :rtype look_pos: np.ndarray
        :param up: the world direction vector corresponding to the up direction in the camera's image_safe_zone. Shape (3)
        :type up: np.ndarray
        :param res: a 2-tuple containing (xres, yres)- the resolution of the image_safe_zone in the x and y direction, in pixels
        :type res: (int, int)
        :param hfov: the horizontal field of view of the camera, in degrees
        :type hfov: float
        :return: the created camera instance
        :rtype: SceneCamera
        """
        r = cvmaths.lookpos_to_rotation_matrix(pos, lookpos, up)
        return SceneCamera(pos, r, res, hfov, name=name, optical_center=optical_center,
                           distortion_coeffs=distortion_coeffs, safe_zone=safe_zone)

    @staticmethod
    def create_camera_from_euler_angles(pos: NDArray, euler_angles: NDArray,
                                        res: Tuple[int, int], hfov: float, name = "",
                                        optical_center: Tuple[float, float] = None,
                                        distortion_coeffs: NDArray = np.zeros(5), safe_zone: int = 100) -> SceneCamera:
        """
        Creates a camera from a lookpos, as opposed to a 3x3 rotation matrix.

        :param pos: the position of the camera in world coordinates. Shape (3)
        :type pos: np.ndarray
        :param euler_angles: an array of length 3, containing the euler angles, in radians
        :type euler_angles: np.ndarray
        :param res: a 2-tuple containing (xres, yres)- the resolution of the image_safe_zone in the x and y direction, in pixels
        :type res: (int, int)
        :param hfov: the horizontal field of view of the camera, in degrees
        :type hfov: float
        :return: the created camera instance
        :rtype: SceneCamera
        """
        r = cvmaths.euler_angles_to_rotation_matrix(euler_angles)
        return SceneCamera(pos, r, res, hfov, name=name, optical_center=optical_center,
                           distortion_coeffs=distortion_coeffs, safe_zone=safe_zone)
