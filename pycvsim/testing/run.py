import numpy as np
from pycvsim.scene_objects.checkerboard_target import CheckerbordTarget
from pycvsim.scene_objects.scene_object import SceneObject
from pycvsim.rendering.scene_offscreen_renderer import SceneOffscreenRenderer
from pycvsim.rendering.scene_camera import SceneCamera
from panda3d.core import GraphicsBuffer
import matplotlib.pyplot as plt


cameras = [
    SceneCamera.create_camera_from_euler_angles(pos=np.array([0.0, 0.0, -1.5]),
                                                euler_angles=np.array([0, 0, 0]),
                                                res=(640, 512), hfov=30.0)
]

renderer = SceneOffscreenRenderer(cameras=cameras)
win: GraphicsBuffer = renderer.win
print(type(win))