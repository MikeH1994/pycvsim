import numpy as np
from pycvsim.sceneobjects.targets.knifeedgetarget import KnifeEdgeTarget
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.core.image_utils import overlay_points_on_image
from pycvsim.routines.knifeedge.knifeedgeroutine import KnifeEdgeRoutine
import matplotlib.pyplot as plt
from pycvsim.routines.siemensstar import SiemensStarRoutine, SiemensStar

camera = SceneCamera(pos=np.array([0.0, 0.0, -1.5]), res=(800, 800), hfov=35.0)
target = SiemensStar(radius=1, n_spokes=10)
routine = SiemensStarRoutine(camera, target)
routine.run()
