from typing import List
import cv2
import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from pycvsim.sceneobjects.sceneobject import SceneObject
from .scenecamera import SceneCamera


class BaseRenderer:
    def __init__(self, cameras: List[SceneCamera] = None, objects: List[SceneObject] = None):
        cameras = cameras if cameras is not None else []
        objects = objects if objects is not None else []
        self.objects: List[SceneObject] = []
        self.cameras: List[SceneCamera] = []
        for camera in cameras:
            self.add_camera(camera)
        for obj in objects:
            self.add_object(obj)

    def add_camera(self, camera: SceneCamera):
        self.cameras.append(camera)

    def remove_camera(self, camera_index: int):
        self.cameras.pop(camera_index)

    def remove_all_cameras(self):
        self.cameras = []

    def remove_object(self, object_index):
        self.objects[object_index].node_path.removeNode()
        self.objects.pop(object_index)

    def add_object(self, obj: SceneObject):
        self.objects.append(obj)

    def remove_all_objects(self):
        while len(self.objects) > 0:
            self.remove_object(0)

    def set_camera_fov(self, camera_index: int, fov: float):
        raise Exception("ahhhhhh!")

    def set_camera_position(self, camera_index: int, pos: NDArray):
        self.cameras[camera_index].pos = pos

    def set_camera_euler_angles(self, camera_index: int, euler_angles: NDArray, degrees=True):
        self.cameras[camera_index].set_euler_angles(euler_angles, degrees=degrees)

    def set_camera_lookpos(self, camera_index: int, lookpos: NDArray, up: NDArray):
        self.cameras[camera_index].set_lookpos(lookpos, up)

    def raycast_scene(self, camera_index):
        camera = self.cameras[camera_index]
        rays = camera.generate_rays().reshape((camera.yres, camera.xres, 6))
        raycasting_scene = o3d.t.geometry.RaycastingScene()
        for obj in self.objects:
            raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(obj.mesh()))
        ans = raycasting_scene.cast_rays(o3d.core.Tensor(rays))
        object_ids: NDArray = ans['geometry_ids'].numpy()
        mask = (object_ids != raycasting_scene.INVALID_ID).astype(np.uint8)
        t_hit = ans['t_hit'].numpy()
        t_hit[mask == 0] = np.inf
        p_hit = camera.pos + t_hit.reshape((camera.yres, camera.xres, 1))*rays[:, :, 3:]
        return {
            "t_hit": t_hit,
            "p_hit": p_hit,
            "object_ids": object_ids,
            "mask": mask
        }

    def deproject_points_on_to_image(self, camera_index: int, object_points: NDArray,
                                     image: NDArray = None) -> NDArray:
        if camera_index >= len(self.cameras):
            raise Exception("Camera index passed is out of bounds")
        camera = self.cameras[camera_index]
        if image is None:
            image = np.zeros((camera.yres, camera.xres, 3), dtype=np.uint8)
        if image.shape[:2] != (camera.yres, camera.xres):
            raise Exception("image_safe_zone does not match resolution of camera index supplied")
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def render_all_images(self, apply_distortion=True) -> List[NDArray]:
        images = []
        for i in range(len(self.cameras)):
            images.append(self.render(i, apply_distortion=apply_distortion))
        return images

    def render(self, camera_index, apply_distortion=False, apply_noise=False, apply_dof=True, return_as_8_bit=True, **kwargs):
        """

        :param camera_index:
        :param apply_distortion:
        :param apply_noise:
        :param apply_dof:
        :param kwargs:
        :return:
        """
        if camera_index >= len(self.cameras):
            raise Exception("Camera index {} is out of bounds".format(camera_index))
        camera = self.cameras[camera_index]
        image = self._render_(camera, return_as_8_bit=return_as_8_bit, **kwargs)

        if apply_dof and camera.dof_model is not None:
            depth = self.raycast_scene(camera_index, apply_distortion=False)["depth"]
            # image = camera.dof_model.apply(image, depth_map=depth, focus_distance=np.percentile(depth.reshape(-1), 50.0))

        if apply_distortion:
            pass # image = camera.distortion_model.distort_image(image, remove_safe_zone=False)

        if apply_noise and camera.noise_model is not None:
            pass # image = camera.noise_model.apply(image)

        if camera.safe_zone > 0:
            safe_zone = camera.safe_zone
            image = image[safe_zone:-safe_zone, safe_zone:-safe_zone]

        return image

    def _render_(self, camera: SceneCamera, return_as_8_bit=True, **kwargs) -> NDArray:
        """
        The base function for the renderer to implement. Returns a 3 channel floating point image, of the
        shape (h, w, 3), where all values lie within the range [0, 1]
        :param camera: the camera that we are using to render
        :param kwargs:
        :return:
        """
        raise Exception("Base function _render_function called")