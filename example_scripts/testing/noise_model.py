import numpy as np
from pycvsim.sceneobjects.targets.checkerboardtarget import CheckerbordTarget
from pycvsim.rendering.open3drenderer import Open3DRenderer
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.core.image_utils import overlay_points_on_image
from pycvsim.optics.noisemodel import NoiseModel
import matplotlib.pyplot as plt

mesh = CheckerbordTarget((9, 8), (0.05, 0.05), board_thickness=0.02, color_bkg=(128, 0, 0), board_boundary=0.05)

noise_model = NoiseModel(image_size=(1024, 1600), preset="default")
cameras = [SceneCamera(pos=np.array([0.0, 0.0, -1.5]), res=(1600, 1024), hfov=30.0, noise_model=noise_model)]

panda3d_renderer = Panda3DRenderer(cameras=cameras, objects=[mesh])

fig = plt.figure()
ax_1 = fig.add_subplot(111)
im_1 = ax_1.imshow(np.zeros((1024, 1600), dtype=np.uint8))
plt.ion()

while True:
    for theta in np.linspace(-180, 180, 100):
        img = panda3d_renderer.render_image(0, apply_noise=True)
        im_1.set_data(img)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.02)
