import numpy as np
import panda3d.core
from pycvsim.targets.checkerboardtarget import CheckerbordTarget
from pycvsim.rendering.open3drenderer import Open3DRenderer
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.camera.basecamera import BaseCamera
import matplotlib.pyplot as plt


mesh = CheckerbordTarget((9, 8), (0.05, 0.05), board_thickness=0.02, color_bkg=(128, 0, 0), board_boundary=0.05)

cameras = [BaseCamera(pos=np.array([0.0, 0.0, -1.5]), res=(1600, 1024), hfov=30.0)]

o3d_renderer = Open3DRenderer(cameras=cameras, objects=[mesh])
panda3d_renderer = Panda3DRenderer(cameras=cameras, objects=[mesh])

mesh.set_euler_angles([0, 0, -38])
image_points = o3d_renderer.cameras[0].get_pixel_point_lies_in(mesh.get_boundary_region())
img_1 = o3d_renderer.render(0, n_samples=32)
img_2 = panda3d_renderer.render(0, n_samples=32, antialiasing=panda3d.core.AntialiasAttrib.MNone)

plt.imshow(img_1)
plt.figure()
plt.imshow(img_2)
plt.figure()
plt.imshow(img_2-img_1)
plt.show()