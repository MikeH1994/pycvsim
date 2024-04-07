from pycvsim.sceneobjects.calibrationtargets import CalibrationTarget
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.rendering.scenerenderer import SceneRenderer
import numbers
import numpy as np
from pycvsim.datageneration.setpoint import ObjectSetpoint, ObjectPosSetpoint, Setpoint, CameraSetpoint
from typing import List
from numpy.typing import NDArray


class CalibrationManager:
    def __init__(self, cameras: List[SceneCamera], calibration_target: CalibrationTarget, **kwargs):
        self.renderer = SceneRenderer(cameras=cameras, objects=[calibration_target])
        self.calibration_target: CalibrationTarget = calibration_target
        self.camera_mode: str = "fixed"
        self.max_rotation: float = 45.0
        self.distance: float = 3.0
        self.n_horizontal: int = 4
        self.n_vertical: int = 4
        self.n_rotation: int = 3
        self.target_fill = 0.25
        self.safe_zone = 25

        if "camera_mode" in kwargs:
            assert(isinstance(self.camera_mode, str))
            self.camera_mode = kwargs["camera_mode"]
        if "max_rotation" in kwargs:
            assert(isinstance(self.max_rotation, numbers.Number))
            self.camera_mode = kwargs["max_rotation"]
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

    def get_closest_position(self, camera: SceneCamera, distance_to_target: float, x_desired: float, y_desired: float,
                             euler_angles: NDArray):
        self.calibration_target.set_pos(np.array([0.0, 0.0, 0.0]))
        self.calibration_target.set_euler_angles(euler_angles)

        x = x_desired
        y = y_desired

        pixel_direction = camera.get_pixel_direction(x, y)
        target_posn = camera.pos + pixel_direction*distance_to_target

        self.calibration_target.set_pos(target_posn)
        x_min = self.safe_zone - 1
        x_max = camera.xres - 1 - self.safe_zone
        y_min = self.safe_zone - 1
        y_max = camera.yres - 1 - self.safe_zone

        object_points = self.calibration_target.get_object_points()
        pixel_coords = camera.get_pixel_point_lies_in(object_points)
        n_attempts = 0
        while not np.all((x_min < pixel_coords[:, 0] < x_max) & (y_min < pixel_coords[:, 1] < y_max)):
            if n_attempts > 5:
                raise Exception("Failed- ")

            if np.any(pixel_coords[:, 0] < x_min) and np.any(pixel_coords[:, 0] > x_max):
                raise Exception("Failed- ")
            if np.any(pixel_coords[:, 1] < y_min) and np.any(pixel_coords[:, 1] > y_max):
                raise Exception("Failed- ")
            dx, dy = 0, 0
            if np.any(pixel_coords[:, 0] < x_min):
                dx = x_min - np.min(pixel_coords[:, 0])
            if np.any(pixel_coords[:, 0] > x_max):
                dx = np.max(pixel_coords[:, 0]) - x_max
            if np.any(pixel_coords[:, 1] < y_min):
                dy = y_min - np.min(pixel_coords[:, 1])
            if np.any(pixel_coords[:, 1] > y_max):
                dy = np.max(pixel_coords[:, 1]) - y_max
            x += dx
            y += dy
            pixel_direction = camera.get_pixel_direction(x, y)
            target_posn = camera.pos + pixel_direction*distance_to_target
            self.calibration_target.set_pos(target_posn)

            object_points = self.calibration_target.get_object_points()
            pixel_coords = camera.get_pixel_point_lies_in(object_points)

            n_attempts += 1
        return target_posn

    def get_corner_positions(self, camera, calibration_target, distance_to_target, euler_angles):
        return [
            self.get_closest_position(camera, calibration_target, distance_to_target, 0.0, 0.0, euler_angles),
            self.get_closest_position(camera, calibration_target, distance_to_target, 1.0, 0.0, euler_angles),
            self.get_closest_position(camera, calibration_target, distance_to_target, 1.0, 0.0, euler_angles),
            self.get_closest_position(camera, calibration_target, distance_to_target, 1.0, 1.0, euler_angles)
        ]


    def generate_setpoints(self):
        setpoints = []
        for camera in self.renderer.cameras:
            setpoints += self.generate_angled_setpoints(camera)
            setpoints += self.generate_horizontal_setpoints(camera)
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
        boundary_region = self.calibration_target.get_boundary_region()
        top_left = np.min(boundary_region, axis=0)
        setpoints = []
        xres = camera.xres
        yres = camera.yres
        distance_to_target, fill_x, fill_y = self.get_desired_distance_to_target(camera)

        dx_px = (1.0 - 2 * (self.safe_zone/xres) - fill_x) / (self.n_horizontal - 1) if self.n_horizontal > 1 else 0.0
        dy_px = (1.0 - 2 * (self.safe_zone/yres) - fill_y) / (self.n_vertical - 1) if self.n_vertical > 1 else 0.0

        for i in range(self.n_horizontal):
            for j in range(self.n_vertical):
                # calculate coordinates of top left of calibration target
                x = self.safe_zone + i * dx_px * camera.xres
                y = self.safe_zone + j * dy_px * camera.yres

                pixel_direction = camera.get_pixel_direction(x, y)
                translation = camera.pos + distance_to_target * pixel_direction - top_left
                alpha = np.random.uniform(-10, 10, 1).item()
                beta = np.random.uniform(-10, 10, 1).item()
                setpoints.append(ObjectPosSetpoint(pos=translation, euler_angles=np.array([alpha, beta, 0.0])))
        return setpoints

    def generate_angled_setpoints(self, camera: SceneCamera):
        self.calibration_target.set_pos(np.array([0.0, 0.0, 0.0]))
        self.calibration_target.set_euler_angles(np.array([0.0, 0.0, 0.0]))
        center = self.calibration_target.get_center()

        setpoints = []
        distance_to_target, fill_x, fill_y = self.get_desired_distance_to_target(camera)
        theta = 25
        # calculate coordinates of top left of calibration target
        dx = (fill_x*np.cos(theta) + fill_y*np.cos(90-theta)) / 2
        dy = (fill_x*np.cos(theta) + fill_y*np.cos(90-theta)) / 2

        min_x = dx*camera.xres + 5*self.safe_zone
        max_x = (1.0 - dx)*camera.xres - 5*self.safe_zone
        min_y = dy*camera.yres + 5*self.safe_zone
        max_y = (1.0 - dy)*camera.yres - 5*self.safe_zone

        for x in np.linspace(min_x, max_x, self.n_rotation):
            for y in np.linspace(min_y, max_y, self.n_rotation):
                for theta in [theta, -theta]:
                    pixel_direction = camera.get_pixel_direction(x, y)
                    translation = camera.pos + distance_to_target * pixel_direction - center

                    alpha = np.random.uniform(-10, 10, 1).item()
                    beta = np.random.uniform(-10, 10, 1).item()
                    euler_angles = np.array([alpha, beta, theta])
                    setpoints.append(ObjectPosSetpoint(pos=translation, euler_angles=euler_angles))
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
