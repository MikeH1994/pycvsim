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
        # loadPrcFileData("", "win-size {} {}".format(xres, yres))
        super().__init__(windowType='offscreen')
        cameras = cameras if cameras is not None else []
        self.objects: List[SceneObject] = []
        self.node_path: NodePath = NodePath()
        self.multiple_samples = multiple_samples
        self.antialiasiang = antialiasing
        self.cameras: List[SceneCamera] = []
        for camera in cameras:
            self.add_camera(camera)

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

    def set_camera_fov(self, hfov):
        self.camLens.setFov(hfov)

    def set_camera_position(self, pos):
        x, y, z = pos
        self.camera.setPos(x, y, z)

    def set_camera_lookpos(self, pos: NDArray, up: NDArray):
        pos = LPoint3f(pos[0], pos[1], pos[2])
        up = LPoint3f(up[0], up[1], up[2])
        self.camera.lookAt(pos, up)

    def set_camera_euler_angles(self, angles):
        self.camera.setHpr(angles[0], angles[1], angles[2])

    def render_image(self, camera_index):
        if camera_index >= len(self.cameras):
            raise Exception("Camera index {} is out of bounds".format(camera_index))
        camera = self.cameras[camera_index]
        self.set_camera_fov(camera.hfov)
        self.set_camera_position(camera.pos)
        self.set_camera_lookpos(camera.lookpos(), camera.up())

        fb_prop = FrameBufferProperties()
        fb_prop.setRgbColor(True)
        fb_prop.setRgbaBits(8, 8, 8, 0)
        fb_prop.setMultisamples(self.multiple_samples)
        fb_prop.setDepthBits(32)
        win_prop = WindowProperties.size(camera.xres, camera.yres)
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

        self.graphicsEngine.removeWindow(self.graphicsEngine.windows[1])
        return bgr_img

    def render_image_from_each_camera(self):
        images = []
        for camera_index in range(len(self.cameras)):
            image = self.render_image(camera_index)
            images.append(image)
        return images

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
            raise Exception("image does not match resolution of camera index supplied")
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
