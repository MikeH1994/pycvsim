import numpy as np
from pycvsim.targets.checkerboardtarget import CheckerbordTarget
from pycvsim.rendering.open3drenderer import Open3DRenderer
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.camera.basecamera import BaseCamera
from pycvsim.core.image_utils import overlay_points_on_image
import matplotlib.pyplot as plt

mesh = CheckerbordTarget((9, 8), (0.05, 0.05), board_thickness=0.02, color_bkg=(128, 0, 0), board_boundary=0.05)

cameras = [BaseCamera(pos=np.array([0.0, 0.0, -1.5]), res=(1600, 1024), hfov=30.0)]

o3d_renderer = Open3DRenderer(cameras=cameras, objects=[mesh])
panda3d_renderer = Panda3DRenderer(cameras=cameras, objects=[mesh])

fig = plt.figure()
ax_1 = fig.add_subplot(121)
ax_2 = fig.add_subplot(122)
im_1 = ax_1.imshow(np.zeros((1600, 1024), dtype=np.uint8))
im_2 = ax_2.imshow(np.zeros((1600, 1024), dtype=np.uint8))
plt.ion()

while True:
    for theta in np.linspace(-180, 180, 100):
        mesh.set_euler_angles([0, 0, theta])
        image_points = o3d_renderer.cameras[0].get_pixel_point_lies_in(mesh.get_boundary_region(), apply_distortion=False)
        img_1 = o3d_renderer.render(0)
        img_1 = overlay_points_on_image(img_1, image_points, radius=6)
        img_2 = panda3d_renderer.render(0)
        img_2 = overlay_points_on_image(img_2, image_points, radius=6)
        im_1.set_data(img_1)
        im_2.set_data(img_2)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.02)
