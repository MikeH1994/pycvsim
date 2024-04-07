from typing import List
import cv2
import time
import numpy as np
import open3d as o3d
from direct.showbase.ShowBase import ShowBase
from numpy.typing import NDArray
from panda3d.core import LPoint3f, GraphicsBuffer, DisplayRegion, LVecBase4f
from panda3d.core import NodePath, AntialiasAttrib, FrameBufferProperties, GraphicsPipe, Texture, GraphicsOutput
from panda3d.core import WindowProperties
from pycvsim.sceneobjects.sceneobject import SceneObject
from .scenecamera import SceneCamera


class SceneRenderer(ShowBase):
    def __init__(self, cameras: List[SceneCamera] = None, objects: List[SceneObject] = None,
                 multiple_samples=32, antialiasing=AntialiasAttrib.MAuto):
        super().__init__(windowType='offscreen')
        cameras = cameras if cameras is not None else []
        objects = objects if objects is not None else []
        self.objects: List[SceneObject] = []
        self.node_path: NodePath = NodePath()
        self.multiple_samples = multiple_samples
        self.antialiasiang = antialiasing
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

    def add_object(self, obj: SceneObject):
        self.objects.append(obj)
        obj.node_path.reparentTo(self.render)

    def remove_object(self, object_index):
        self.objects[object_index].node_path.removeNode()
        self.objects.pop(object_index)

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

    def set_render_camera_fov(self, hfov, vfov):
        self.camLens.setFov(hfov, vfov)

    def set_render_camera_position(self, pos):
        x, y, z = pos
        self.camera.setPos(x, y, z)

    def set_render_camera_lookpos(self, pos: NDArray, up: NDArray):
        pos = LPoint3f(pos[0], pos[1], pos[2])
        up = LPoint3f(up[0], up[1], up[2])
        self.camera.lookAt(pos, up)

    def render_all_images(self, apply_distortion=True, remove_safe_zone=True) -> List[NDArray]:
        images = []
        for i in range(len(self.cameras)):
            images.append(self.render_image(i, apply_distortion=apply_distortion, remove_safe_zone=remove_safe_zone))
        return images

    def render_image(self, camera_index, apply_distortion=True, remove_safe_zone=True):
        if camera_index >= len(self.cameras):
            raise Exception("Camera index {} is out of bounds".format(camera_index))
        camera = self.cameras[camera_index]
        self.set_render_camera_fov(*camera.get_fov(include_safe_zone=apply_distortion))
        self.set_render_camera_position(camera.pos)
        self.set_render_camera_lookpos(camera.get_lookpos(), camera.get_up())
        xres, yres = camera.get_res(include_safe_zone=apply_distortion)

        fb_prop = FrameBufferProperties()
        fb_prop.setRgbColor(True)
        fb_prop.setRgbaBits(8, 8, 8, 0)
        fb_prop.setMultisamples(self.multiple_samples)
        fb_prop.setDepthBits(32)
        win_prop = WindowProperties.size(xres, yres)
        window: GraphicsBuffer = self.graphicsEngine.makeOutput(self.pipe, "cameraview", 0, fb_prop, win_prop,
                                                                GraphicsPipe.BFRefuseWindow, self.win.getGsg(), self.win)
        disp_region: DisplayRegion = window.makeDisplayRegion()
        disp_region.setCamera(self.cam)
        bgr_tex = Texture()
        window.addRenderTexture(bgr_tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)
        window.set_clear_color(LVecBase4f(0.2, 0.2, 0.2, 1.0))

        for obj in self.objects:
            obj.node_path.setAntialias(self.antialiasiang)

        self.graphicsEngine.renderFrame()
        bgr_img = np.frombuffer(bgr_tex.getRamImage(), dtype=np.uint8)
        bgr_img.shape = (bgr_tex.getYSize(), bgr_tex.getXSize(), bgr_tex.getNumComponents())
        bgr_img = bgr_img[:, ::-1, :3]
        if apply_distortion:
            bgr_img = camera.distortion_model.distort_image(bgr_img, remove_safe_zone=remove_safe_zone)
        self.graphicsEngine.removeWindow(self.graphicsEngine.windows[1])
        img = np.zeros(bgr_img.shape, dtype=np.uint8)
        img[:, :, :] = bgr_img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def raycast_scene(self, camera_index):
        rays = self.cameras[camera_index].generate_rays()
        raycasting_scene = o3d.t.geometry.RaycastingScene()
        for obj in self.objects:
            raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(obj.mesh()))
        ans = raycasting_scene.cast_rays(o3d.core.Tensor(rays))
        return {
            "t_hit": ans['t_hit'].numpy(),
            'object_ids': ans['geometry_ids'].numpy().astype(np.int32)
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
