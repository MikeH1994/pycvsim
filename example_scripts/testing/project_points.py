import numpy as np
from pycvsim.sceneobjects.calibrationtargets.checkerboardtarget import CheckerbordTarget
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.scenerenderer import SceneRenderer
from pycvsim.rendering.scenecamera import SceneCamera
import matplotlib.pyplot as plt
import cv2

obj = CheckerbordTarget((7, 7), (0.05, 0.05), board_thickness=0.02,
                        color_1=(255, 255, 255), color_2=(0, 0, 0),
                        color_bkg=(128, 0, 0), board_boundary=0.05, name="checkerboard")
cameras = [SceneCamera(pos=np.array([0.0, 0.0, -2.0]), res=(720, 720), hfov=30.0, safe_zone=0)]
renderer = SceneRenderer(cameras=cameras, objects=[obj])
obj.set_euler_angles(np.array([0, 0, 20.0]))


img_render = renderer.render_image(0, apply_distortion=True)
plt.imshow(img_render)
plt.show()
