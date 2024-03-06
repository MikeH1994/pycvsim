from panda3d.core import NodePath, AntialiasAttrib, LVecBase2i, FrameBufferProperties, GraphicsPipe, Texture, GraphicsOutput
from panda3d.core import Camera, PerspectiveLens
from direct.showbase.ShowBase import ShowBase
import numpy as np
from pycvsim.scene_objects.scene_object import SceneObject
from panda3d.core import loadPrcFileData
from panda3d.core import WindowProperties

from typing import List
import open3d as o3d
from .scene_camera import SceneCamera


class SceneOffscreenRenderer(ShowBase):
    def __init__(self, xres=640, yres=512, cameras: List[SceneCamera] = None):
        loadPrcFileData("", "win-size {} {}".format(xres, yres))
        super().__init__(windowType='offscreen')
        self.objects = []
        self.setBackgroundColor(0, 0, 0)
        self.node_path = NodePath()
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.cameras = cameras if cameras is not None else []


    def add_object(self, object: SceneObject):
        self.objects.append(object)
        object.node_path.reparentTo(self.render)

    def set_camera_fov(self, hfov):
        self.camLens.setFov(hfov)

    def set_camera_position(self, pos):
        x, y, z = pos
        self.camera.setPos(x, y, z)

    def set_camera_lookpos(self, pos):
        x, y, z = pos
        self.camera.lookAt(x, y, z)

    def set_camera_look_at(self, node_path):
        self.camera.lookAt(node_path)

    def set_camera_quaternion(self, r):
        self.camera.setQuat(r)

    def set_camera_euler_angles(self, angles):
        self.camera.setHpr(angles[0], angles[1], angles[2])

    def set_resolution(self, xres, yres):
        """
        properties = WindowProperties()
        properties.setSize(xres, yres)
        # self.win.requestProperties(properties)
        """

        # this does not work
        """
        self.win.size = LVecBase2i(xres, yres)
        self.win.fb_size = LVecBase2i(xres, yres)
        self.win.sbs_left_size = LVecBase2i(xres, yres)
        self.win.sbs_right_size = LVecBase2i(xres, yres)
        """

        """"
        #Cannot resize buffer unless it is created with BF_resizeable flag
        self.win.set_size(xres, yres)
        """

    def render_image(self, camera_index):
        if camera_index >= len(self.cameras):
            raise Exception("Camera index {} is out of bounds".format(camera_index))

        camera = self.cameras[camera_index]
        self.set_camera_fov(camera.hfov)
        self.set_camera_position(camera.pos)
        self.set_camera_lookpos(camera.look_pos)

        fb_prop = FrameBufferProperties()
        fb_prop.setRgbColor(True)
        fb_prop.setRgbaBits(8, 8, 8, 0)
        fb_prop.setDepthBits(24)
        win_prop = WindowProperties.size(camera.x_res, camera.y_res)
        window = self.graphicsEngine.makeOutput(self.pipe, "cameraview", 0, fb_prop, win_prop,
                                                GraphicsPipe.BFRefuseWindow)
        disp_region = window.makeDisplayRegion()

        disp_region.setCamera(self.cam)
        bgr_tex = Texture()
        window.addRenderTexture(bgr_tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)
        self.graphicsEngine.renderFrame()

        bgr_img = np.frombuffer(bgr_tex.getRamImage(), dtype=np.uint8)
        bgr_img.shape = (bgr_tex.getYSize(), bgr_tex.getXSize(), bgr_tex.getNumComponents())

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
            raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(obj.mesh))
        depth_image = raycasting_scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        #mesh_id_image[mesh_id_image != raycasting_scene.INVALID_ID] = 1
        return depth_image


"""
        # Create frame buffer properties
        fb_prop = FrameBufferProperties()
        fb_prop.setRgbColor(True)
        # Only render RGB with 8 bit for each channel, no alpha channel
        fb_prop.setRgbaBits(8, 8, 8, 0)
        fb_prop.setDepthBits(24)
        # Create window properties
        win_prop = WindowProperties.size(xres, yres)
        # Create window (offscreen)
        window = self.graphicsEngine.makeOutput(self.pipe, "cameraview_1", 0, fb_prop, win_prop,
                                                GraphicsPipe.BFRefuseWindow)
        camera_obj = Camera("Camera", PerspectiveLens())

        self.graphicsEngine.remove_all_windows()
        self.graphicsEngine.addWindow(window, 0)
        self.win = window

"""



"""
IMG_W = 640
IMG_H = 480
base = ShowBase(fStartDirect=True, windowType='offscreen')
fb_prop = FrameBufferProperties()
fb_prop.setRgbColor(True)
fb_prop.setRgbaBits(8, 8, 8, 0)
fb_prop.setDepthBits(24)
win_prop = WindowProperties.size(IMG_W, IMG_H)
window = base.graphicsEngine.makeOutput(base.pipe, "cameraview", 0, fb_prop, win_prop, GraphicsPipe.BFRefuseWindow)
disp_region = window.makeDisplayRegion()
disp_region.setCamera(cam_obj)
bgr_tex = Texture()
window.addRenderTexture(bgr_tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)
bgr_img = np.frombuffer(bgr_tex.getRamImage(), dtype=np.uint8)
bgr_img.shape = (bgr_tex.getYSize(), bgr_tex.getXSize(), bgr_tex.getNumComponents())
"""