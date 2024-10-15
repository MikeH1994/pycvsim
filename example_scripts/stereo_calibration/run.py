import numpy as np
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.camera.basecamera import BaseCamera
from pycvsim.routines.stereophotogrammetry.stereoroutine import StereoRoutine

camera_1 = BaseCamera.create_camera_from_lookpos(pos=np.array([0.0, 0.0, -2.5]), lookpos=np.array([0.0, 0.0, 0.0]),
                                                 up=np.array([0.0, 1.0, 0.0]), res=(1001, 1001), hfov=35.0)
camera_2 = BaseCamera.create_camera_from_lookpos(pos=np.array([-0.3, 0.0, -2.5]), lookpos=np.array([0.0, 0.0, 0.0]),
                                                 up=np.array([0.0, 1.0, 0.0]), res=(1001, 1001), hfov=35.0)
routine = StereoRoutine(camera_1, camera_2)
mesh = SceneObject.load_armadillo()
routine.run(mesh)
