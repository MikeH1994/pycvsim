from pycvsim.targets.calibrationtarget import CalibrationTarget
from pycvsim.camera.basecamera import BaseCamera
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
import numbers
import numpy as np
from pycvsim.routines.calibration.setpoint import ObjectSetpoint, ObjectPosSetpoint, Setpoint, CameraSetpoint
from typing import List
from numpy.typing import NDArray


class ImageSetGenerator:
    def __init__(self, cameras: List[BaseCamera], calibration_target: CalibrationTarget, **kwargs):
        self.renderer = Panda3DRenderer(cameras=cameras, objects=[calibration_target])
        self.calibration_target: CalibrationTarget = calibration_target
        self.random_alpha: float = 20
        self.random_beta: float = 20
        self.random_gamma: float = 45.0
        self.random_distance: float = 0.4
        self.camera_mode: str = "fixed"
        self.distance: float = 3.0
        self.n_horizontal: int = 4
        self.n_vertical: int = 4
        self.n_rotation: int = 3
        self.distance_to_target = 2.5
        self.safe_zone = 25

        if "camera_mode" in kwargs:
            assert(isinstance(self.camera_mode, str))
            self.camera_mode = kwargs["camera_mode"]
        if "n_vertical" in kwargs:
            self.n_vertical = int(kwargs["n_vertical"])
        if "n_horizontal" in kwargs:
            self.n_horizontal = int(kwargs["n_horizontal"])
        if "n_rotation" in kwargs:
            self.n_rotation = int(kwargs["n_rotation"])
        if "safe_zone" in kwargs:
            self.safe_zone = int(kwargs["safe_zone"])
        if "target_fill" in kwargs:
            self.target_fill = float(kwargs["target_fill"])
        if "random_alpha" in kwargs:
            self.random_alpha = float(kwargs["random_alpha"])
        if "random_beta" in kwargs:
            self.random_beta = float(kwargs["random_beta"])
        if "random_gamma" in kwargs:
            assert(isinstance(self.random_gamma, numbers.Number))
            self.camera_mode = kwargs["random_gamma"]

    def get_closest_position(self, camera: BaseCamera, distance_to_target: float, x_desired: float, y_desired: float,
                             euler_angles: NDArray):
        self.calibration_target.set_pos(np.array([0.0, 0.0, 0.0]))
        self.calibration_target.set_euler_angles(euler_angles)

        x = x_desired
        y = y_desired
        x_min = self.safe_zone - 1
        x_max = camera.xres - 1 - self.safe_zone
        y_min = self.safe_zone - 1
        y_max = camera.yres - 1 - self.safe_zone

        target_posn = camera.get_3d_point_from_pixel(x, y, distance_to_target)

        self.calibration_target.set_pos(target_posn)
        object_points = self.calibration_target.get_object_points()
        pixel_coords = camera.get_pixel_point_lies_in(object_points)
        x0, x1 = np.min(pixel_coords[:, 0]), np.max(pixel_coords[:, 0])
        y0, y1 = np.min(pixel_coords[:, 1]), np.max(pixel_coords[:, 1])
        n_attempts = 0
        while not ((x0 > x_min) and (x1 < x_max) and (y0 > y_min) and (y1 < y_max)):
            if n_attempts > 10:
                raise Exception("Failed- ")

            if x0 < x_min and x1 > x_max:
                raise Exception("Failed- ")
            if y0 < y_min and y1 > y_max:
                raise Exception("Failed- ")
            dx, dy = 0, 0
            if x0 < x_min:
                dx = max((x_min - x0)*0.5, 1)
            elif x1 > x_max:
                dx = min((x_max - x1)*0.5, -1)
            if y0 < y_min:
                dy = max((y_min - y0)*0.5, 1)
            elif y1 > y_max:
                dy = min((y_max - y1)*0.5, -1)
            x += dx
            y += dy
            target_posn = camera.get_3d_point_from_pixel(x, y, distance_to_target)
            self.calibration_target.set_pos(target_posn)

            object_points = self.calibration_target.get_object_points()
            pixel_coords = camera.get_pixel_point_lies_in(object_points)
            x0, x1 = np.min(pixel_coords[:, 0]), np.max(pixel_coords[:, 0])
            y0, y1 = np.min(pixel_coords[:, 1]), np.max(pixel_coords[:, 1])
            n_attempts += 1
        return target_posn, x, y

    def generate_setpoints(self):
        setpoints = []
        for camera in self.renderer.cameras:
            setpoints += self.generate_horizontal_setpoints(camera)
            setpoints += self.generate_angled_setpoints(camera)
        return setpoints

    def get_desired_distance_to_target(self, camera):
        xres = camera.xres
        yres = camera.yres
        boundary_region = self.calibration_target.get_boundary_region()
        width = np.max(boundary_region[:, 0]) - np.min(boundary_region[:, 0])
        height = np.max(boundary_region[:, 1]) - np.min(boundary_region[:, 1])
        fill_x = self.target_fill if width/xres > height/yres else self.target_fill * (width / height) * yres / xres
        fill_y = self.target_fill if height/yres > width/xres else self.target_fill * (height / width) * xres / yres
        hfov = camera.hfov
        image_plane_width = width / self.target_fill / 2.0
        distance_to_target = image_plane_width / np.tan(np.radians(hfov / 2.0))
        return distance_to_target, fill_x, fill_y

    def generate_horizontal_setpoints(self, camera) -> List[Setpoint]:
        self.calibration_target.set_pos(np.array([0.0, 0.0, 0.0]))
        self.calibration_target.set_euler_angles(np.array([0.0, 0.0, 0.0]))
        # distance_to_target, _, _ = self.get_desired_distance_to_target(camera)

        setpoints = []
        for i, x in enumerate(np.linspace(self.safe_zone, camera.xres - self.safe_zone, self.n_horizontal)):
            for j, y in enumerate(np.linspace(self.safe_zone, camera.yres - self.safe_zone, self.n_vertical)):
                alpha = np.random.uniform(-self.random_alpha, self.random_alpha, 1).item()
                beta = np.random.uniform(-self.random_beta, self.random_beta, 1).item()
                euler_angles = np.array([alpha, beta, 0.0])
                d = self.random_distance
                distance_to_target = self.distance_to_target + np.random.uniform(-d, d, 1).item()
                target_posn, _, _ = self.get_closest_position(camera, distance_to_target, x, y, euler_angles)
                setpoints.append(ObjectPosSetpoint(pos=target_posn, euler_angles=euler_angles))
        return setpoints

    def generate_angled_setpoints(self, camera: BaseCamera):
        self.calibration_target.set_pos(np.array([0.0, 0.0, 0.0]))
        self.calibration_target.set_euler_angles(np.array([0.0, 0.0, 0.0]))

        distance_to_target = self.distance_to_target # , fill_x, fill_y = self.get_desired_distance_to_target(camera)
        setpoints = []

        _, x_min, y_min = self.get_closest_position(camera, distance_to_target, 0, 0,
                                                    np.array([0.0, 0.0, 0.0]))
        _, x_max, y_max = self.get_closest_position(camera, distance_to_target, camera.xres, camera.yres,
                                                    np.array([0.0, 0.0, 0.0]))
        print(np.linspace(x_min, x_max, self.n_rotation))
        print(np.linspace(y_min, y_max, self.n_rotation))
        for x in np.linspace(x_min, x_max, self.n_rotation):
            for y in np.linspace(y_min, y_max, self.n_rotation):
                alpha = np.random.uniform(-self.random_alpha, self.random_alpha, 1).item()
                beta = np.random.uniform(-self.random_beta, self.random_beta, 1).item()
                gamma = np.random.uniform(-self.random_gamma, self.random_gamma, 1).item()

                euler_angles = np.array([alpha, beta, gamma])
                d = self.distance_to_target + np.random.uniform(-self.random_distance, self.random_distance, 1).item()
                target_posn, x_i, y_i = self.get_closest_position(camera, d, x, y, euler_angles)
                #print("desired: ({},{}) retrieved: ({},{})".format(x, y, x_i, y_i))
                setpoints.append(ObjectPosSetpoint(pos=target_posn, euler_angles=euler_angles))
        return setpoints

    def run(self, setpoints: List[Setpoint]) -> List[List[NDArray]]:
        returned_images = []
        for sp in setpoints:
            if isinstance(sp, ObjectSetpoint):
                sp.apply(self.calibration_target)
            elif isinstance(sp, CameraSetpoint):
                for camera in self.renderer.cameras:
                    sp.apply(camera)
            returned_images.append(self.renderer.render_all_images())
        return returned_images
