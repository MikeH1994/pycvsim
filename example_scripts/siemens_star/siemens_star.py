import numpy as np
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.routines.siemensstar import SiemensStarRoutine, SiemensStar

camera = SceneCamera(pos=np.array([0.0, 0.0, -1.5]), res=(800, 800), hfov=35.0)
target = SiemensStar(radius=1, n_spokes=10)
routine = SiemensStarRoutine(camera, target)
routine.run()
