import numpy as np
from pycvsim.targets.checkerboardtarget import CheckerbordTarget
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.camera.basecamera import BaseCamera
import matplotlib.pyplot as plt
import cv2

mesh, object_points = CheckerbordTarget.create_target((7, 6), (0.05, 0.05), board_thickness=0.02,
                                                      color_bkg=(128, 0, 0), board_boundary=0.05)

cameras = [
    BaseCamera.create_camera_from_euler_angles(pos=np.array([0.0, 0.0, -1.5]),
                                               euler_angles=np.array([0, 0, 0]),
                                               res=(640, 512), hfov=30.0),
]

obj_mesh = SceneObject(mesh)
renderer = Panda3DRenderer(cameras=cameras)
renderer.add_object(obj_mesh)

fig = plt.figure()
ax_1 = fig.add_subplot(121)
ax_2 = fig.add_subplot(122)
im_1 = ax_1.imshow(np.zeros((640, 512), dtype=np.uint8))
im_2 = ax_2.imshow(np.zeros((640, 512), dtype=np.uint8))
plt.ion()

while True:
    for axis in [0, 1, 2]: #
        for theta in np.linspace(-180, 180, 6):
            pos = np.random.uniform(-0.2, 0.2, 3)
            angles = np.zeros(3)
            angles[axis] = theta
            obj_mesh.set_euler_angles(angles)
            obj_mesh.set_pos(pos)
            img_1 = renderer.render(0)
            img_2 = renderer.raycast_scene(0)["object_ids"]
            img_2 = cv2.applyColorMap(((img_2+1)*255).astype(np.uint8), cv2.COLORMAP_JET)

            im_1.set_data(img_1)
            im_2.set_data(img_2)

            im_2.set_data((img_2+1).astype(np.uint8)*255)
            ax_1.set_title("axis {}".format(axis))
            ax_2.set_title("o3d obj index mask")
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.05)