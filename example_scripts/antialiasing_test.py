import numpy as np
from pycvsim.sceneobjects.calibrationtargets.checkerboardtarget import CheckerbordTarget
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.scenerenderer import SceneRenderer
from pycvsim.rendering.scenecamera import SceneCamera
import matplotlib.pyplot as plt
import panda3d.core

mesh, object_points = CheckerbordTarget.create_target((7, 6), (0.05, 0.05), board_thickness=0.02,
                                                      color_bkg=(128, 0, 0), board_boundary=0.05)

cameras = [
    SceneCamera.create_camera_from_lookpos(pos=np.array([0.8, 0.8, -1.5]),
                                           lookpos=np.array([0.0, 0.0, 0.0]),
                                           up=np.array([0.0, 1.0, 0.0]),
                                           res=(640, 512), hfov=50.0)]
obj_mesh = SceneObject(mesh)
renderer = SceneRenderer(cameras=cameras)
renderer.add_object(obj_mesh)

renderer.multiple_samples = 128
renderer.antialiasiang = panda3d.core.AntialiasAttrib.MAuto
img_auto_16 = renderer.render_image(0)

renderer.multiple_samples = 128
renderer.antialiasiang = panda3d.core.AntialiasAttrib.MLine
img_line_16 = renderer.render_image(0)

renderer.multiple_samples = 128
renderer.antialiasiang = panda3d.core.AntialiasAttrib.MMultisample
img_multi_16 = renderer.render_image(0)

renderer.multiple_samples = 0
renderer.antialiasiang = panda3d.core.AntialiasAttrib.MNone
img_none_0 = renderer.render_image(0)

# check which antialiasing works best for checkerboard
fig, ax = plt.subplots(2, 2)
fig.suptitle('plt.subplots')
ax[0][0].imshow(img_auto_16)
ax[0][0].set_title("Auto 16")
ax[0][1].imshow(img_multi_16)
ax[0][1].set_title("Multisample 16")
ax[1][0].imshow(img_line_16)
ax[1][0].set_title("Line 16")
ax[1][1].imshow(img_none_0)
ax[1][1].set_title("None 0")
plt.show()

images = []
for n_samples in [0, 4, 16, 128]:
    renderer.multiple_samples = n_samples
    renderer.antialiasiang = panda3d.core.AntialiasAttrib.MAuto
    images.append(renderer.render_image(0))
# check which antialiasing works best for checkerboard
fig, ax = plt.subplots(2, 2)
fig.suptitle('plt.subplots')
ax[0][0].imshow(images[0])
ax[0][0].set_title("0")
ax[0][1].imshow(images[1])
ax[0][1].set_title("4")
ax[1][0].imshow(images[2])
ax[1][0].set_title("16")
ax[1][1].imshow(images[3])
ax[1][1].set_title("128")
plt.show()
