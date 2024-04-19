import numpy as np
from pycvsim.sceneobjects.targets.checkerboardtarget import CheckerbordTarget
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.core.image_utils import overlay_points_on_image
import matplotlib.pyplot as plt
import cv2

obj = CheckerbordTarget((7, 6), (0.05, 0.05), board_thickness=0.02,
                        color_1=(255, 255, 255), color_2=(0, 0, 0),
                        color_bkg=(128, 0, 0), board_boundary=0.05, name="checkerboard")
cameras = [SceneCamera(pos=np.array([0.0, 0.0, -1.2]), res=(720, 720), hfov=30.0, safe_zone=0)]
renderer = Panda3DRenderer(cameras=cameras, objects=[obj])
obj.set_euler_angles(np.array([0, 0, 20.0]))
object_points = obj.get_object_points()
image_points = renderer.cameras[0].get_pixel_point_lies_in(object_points)

img_render = renderer.render_image(0, apply_distortion=True)
ret, corners = cv2.findChessboardCorners(img_render, (7, 6), None)
img_overlayed = overlay_points_on_image(img_render, image_points)
if ret:
    img_overlayed = cv2.drawChessboardCorners(img_overlayed, (7, 6), corners, ret)

plt.imshow(img_overlayed)
plt.show()
