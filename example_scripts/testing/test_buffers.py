import numpy as np
import time
from pycvsim.sceneobjects.targets.checkerboardtarget import CheckerbordTarget
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.scenecamera import SceneCamera

mesh, object_points = CheckerbordTarget.create_target((7, 6), (0.05, 0.05), board_thickness=0.02,
                                                      color_bkg=(128, 0, 0), board_boundary=0.05)

cameras = [
    SceneCamera.create_camera_from_euler_angles(pos=np.array([0.2, 0.2, -1.5]),
                                                euler_angles=np.array([0, 0, 0]),
                                                res=(1000, 800), hfov=30.0),
    SceneCamera.create_camera_from_lookpos(pos=np.array([0.8, 0.8, -1.5]),
                                           lookpos=np.array([0.0, 0.0, 0.0]),
                                           up=np.array([0.0, 1.0, 0.0]),
                                           res=(640, 512), hfov=50.0),
    SceneCamera.create_camera_from_lookpos(pos=np.array([0.8, 0.8, 0.8]),
                                           lookpos=np.array([0.0, 0.0, 0.0]),
                                           up=np.array([0.0, 1.0, 0.0]),
                                           res=(320, 256), hfov=100.0)]

obj_mesh = SceneObject(mesh)
renderer = Panda3DRenderer(cameras=cameras)
renderer.add_object(obj_mesh)
for i in range(100):
    start = time.time()
    renderer.render(0)
    end = time.time()
    print("time taken = {:.3f}s".format(end-start))
    print(len(renderer.graphicsEngine.windows))