import numpy as np
from pycvsim.sceneobjects.targets.checkerboardtarget import CheckerbordTarget
from pycvsim.rendering.open3drenderer import Open3DRenderer
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.core.image_utils import overlay_points_on_image
import matplotlib.pyplot as plt

mesh = CheckerbordTarget((9, 8), (0.05, 0.05), board_thickness=0.02, color_bkg=(128, 0, 0), board_boundary=0.05)

cameras = [SceneCamera(pos=np.array([0.0, 0.0, -1.5]), res=(800, 800), hfov=30.0)]
renderer = Open3DRenderer(cameras=cameras, objects=[mesh])

fig = plt.figure()
ax_1 = fig.add_subplot(111)
im_1 = ax_1.imshow(np.zeros((800, 800), dtype=np.uint8))
plt.ion()

while True:
    optical_center = np.random.uniform(low=250, high=550, size=2)
    renderer.cameras[0].cx = optical_center[0]
    renderer.cameras[0].cy = optical_center[1]
    img_1 = renderer.render_image(0, n_samples=1)
    im_1.set_data(img_1)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.02)
