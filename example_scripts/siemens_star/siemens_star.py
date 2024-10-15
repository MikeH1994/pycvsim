import numpy as np
from pycvsim.camera.basecamera import BaseCamera
from pycvsim.routines.siemensstar import SiemensStarRoutine, SiemensStar

camera = BaseCamera(pos=np.array([0.0, 0.0, -1.5]), res=(800, 800), hfov=35.0)
target = SiemensStar(radius=1, n_spokes=10)
routine = SiemensStarRoutine(camera, target)
routine.run()
