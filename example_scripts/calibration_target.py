import numpy as np
from pycvsim.sceneobjects.calibrationtargets.checkerboardtarget import CheckerbordTarget
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.scenerenderer import SceneRenderer
from pycvsim.rendering.scenecamera import SceneCamera
import matplotlib.pyplot as plt

mesh, object_points = CheckerbordTarget.create_target((7, 6), (0.05, 0.05), board_thickness=0.02,
                                                      color_bkg=(128, 0, 0), board_boundary=0.05)

cameras = [
    SceneCamera.create_camera_from_euler_angles(pos=np.array([0.0, 0.0, -1.5]),
                                                euler_angles=np.array([0, 0, 0]),
                                                res=(1280, 1024), hfov=30.0)
]

obj_mesh = SceneObject(mesh)
renderer = SceneRenderer(cameras=cameras, objects=[obj_mesh])

fig = plt.figure()
ax_1 = fig.add_subplot(111)
im_1 = ax_1.imshow(np.zeros((640, 512), dtype=np.uint8))
plt.ion()

while True:
    for theta in np.linspace(-180, 180, 100):
        obj_mesh.set_euler_angles([0, theta, 0])
        img_1 = renderer.render_image(0)
        im_1.set_data(img_1)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.02)
