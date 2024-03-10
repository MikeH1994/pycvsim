import numpy as np
from pycvsim.sceneobjects.calibrationtargets.checkerboardtarget import CheckerbordTarget
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.scenerenderer import SceneRenderer
from pycvsim.rendering.scenecamera import SceneCamera
import matplotlib.pyplot as plt

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
renderer = SceneRenderer(cameras=cameras)
renderer.add_object(obj_mesh)

while True:
    for i in range(len(cameras)):
        img_1 = renderer.render_image(i)
        plt.figure()
        plt.imshow(img_1)
        plt.figure()
        img = renderer.raycast_scene(i)["object_ids"]
        plt.imshow(img)
        plt.show()
