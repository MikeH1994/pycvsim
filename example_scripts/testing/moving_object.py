import numpy as np
from pycvsim.targets.checkerboardtarget import CheckerbordTarget
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.scenecamera import SceneCamera
import matplotlib.pyplot as plt

mesh, object_points = CheckerbordTarget.create_target((7, 6), (0.05, 0.05), board_thickness=0.02,
                                                      color_bkg=(128, 0, 0), board_boundary=0.05)

cameras = [
    SceneCamera.create_camera_from_euler_angles(pos=np.array([0.0, 0.0, -1.5]),
                                                euler_angles=np.array([0, 0, 0]),
                                                res=(640, 512), hfov=30.0),
]

obj_mesh = SceneObject(mesh)
renderer = Panda3DRenderer(cameras=cameras)
renderer.add_object(obj_mesh)

ax = plt.subplot(1, 1, 1)
im = ax.imshow(np.zeros((600, 600)))
plt.ion()

while True:
    for x in np.linspace(-0.2, 0.2, 3):
        for y in np.linspace(-0.2, 0.2, 3):
            obj_mesh.set_pos(np.array([x, y, 0.0]))
            img = renderer.render(0)
            im.set_data(img)
            plt.pause(0.05)
