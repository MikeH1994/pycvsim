import numpy as np
from pycvsim.sceneobjects.targets.knifeedgetarget import KnifeEdgeTarget
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.core.image_utils import overlay_points_on_image
import matplotlib.pyplot as plt

cameras = [SceneCamera(pos=np.array([0.0, 0.0, -1.5]), res=(1600, 1600), hfov=35.0)]
renderer = Panda3DRenderer(cameras=cameras)

#for theta in np.linspace(310, 360, 15):
#    KnifeEdgeTarget(0.2, angle=theta)

fig = plt.figure()
ax_1 = fig.add_subplot(111)
im_1 = ax_1.imshow(np.zeros((1600, 1600), dtype=np.uint8))
plt.ion()

while True:
    for theta in np.linspace(0, 360.0, 30):
        mesh = KnifeEdgeTarget(0.5, angle=theta)
        renderer.remove_all_objects()
        renderer.add_object(mesh)
        img_1 = renderer.render(0)
        im_1.set_data(img_1)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.title("{:.3f}".format(theta))
        plt.pause(0.01)
