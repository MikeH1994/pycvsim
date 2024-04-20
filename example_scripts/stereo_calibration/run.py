import numpy as np
from pycvsim.sceneobjects.targets.knifeedgetarget import KnifeEdgeTarget
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.core.image_utils import overlay_points_on_image
from pycvsim.routines.stereophotogrammetry.stereoroutine import StereoRoutine
import matplotlib.pyplot as plt

camera_1 = SceneCamera.create_camera_from_lookpos(pos=np.array([0.2, 0.0, -2.5]), lookpos=np.array([0.0, 0.0, 0.0]),
                                                  up=np.array([0.0, 1.0, 0.0]), res=(1000, 1000), hfov=35.0)
camera_2 = SceneCamera.create_camera_from_lookpos(pos=np.array([-0.2, 0.0, -2.5]), lookpos=np.array([0.0, 0.0, 0.0]),
                                                  up=np.array([0.0, 1.0, 0.0]), res=(1000, 1000), hfov=35.0)
routine = StereoRoutine(camera_1, camera_2)
mesh = SceneObject.load_armadillo()
routine.run(mesh)
