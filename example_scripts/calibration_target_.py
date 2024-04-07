import numpy as np
from pycvsim.sceneobjects.calibrationtargets.checkerboardtarget import CheckerbordTarget
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.scenerenderer import SceneRenderer
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.core.image_utils import overlay_points_on_image
import matplotlib.pyplot as plt

mesh = CheckerbordTarget((9, 8), (0.05, 0.05), board_thickness=0.02, color_bkg=(128, 0, 0), board_boundary=0.05)

cameras = [SceneCamera(pos=np.array([0.0, 0.0, -1.5]), res=(1600, 1024), hfov=30.0)]

renderer = SceneRenderer(cameras=cameras, objects=[mesh])

points = mesh.get_object_points().reshape((9, 8, 3))
image_points = renderer.cameras[0].get_pixel_point_lies_in(points)
print(image_points)
