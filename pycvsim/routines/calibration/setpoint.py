from numpy.typing import NDArray
from typing import Union
from pycvsim.camera.basecamera import BaseCamera
from pycvsim.sceneobjects.sceneobject import SceneObject


class Setpoint:
    def __init__(self):
        pass


class ObjectSetpoint(Setpoint):
    def __init__(self):
        super().__init__()

    def apply(self, scene_object: SceneObject):
        raise Exception("Base function ObjectSetpoint.apply() called")


class ObjectPosSetpoint(ObjectSetpoint):
    pos: Union[NDArray, None]
    euler_angles: Union[NDArray, None]
    pos_mode: str
    angle_mode: str

    def __init__(self, pos: Union[NDArray, None] = None,
                 euler_angles: Union[NDArray, None] = None,
                 pos_mode: str = "absolute", angle_mode: str = "absolute"):
        super().__init__()
        self.pos = pos
        self.euler_angles = euler_angles
        self.pos_mode = pos_mode
        self.angle_mode = angle_mode

    def apply(self, scene_object: SceneObject):
        if self.pos is not None:
            scene_object.set_pos(self.pos, mode=self.pos_mode)

        if self.euler_angles is not None:
            scene_object.set_euler_angles(self.euler_angles, self.angle_mode)


class CameraSetpoint(Setpoint):
    def __init__(self):
        super().__init__()

    def apply(self, camera: BaseCamera):
        raise Exception("Base function CameraSetpoint.apply() called")


class CameraLookPosSetpoint(CameraSetpoint):
    pos: Union[NDArray, None]
    lookpos: NDArray
    up: NDArray

    def __init__(self, pos: Union[NDArray, None], lookpos: NDArray, up: NDArray):
        super().__init__()
        self.pos = pos
        self.lookpos = lookpos
        self.up = up

    def apply(self, camera: BaseCamera):
        if self.pos is not None:
            camera.pos = self.pos
        camera.set_lookpos(self.lookpos, self.up)


class CameraPosSetpoint(CameraSetpoint):
    pos: NDArray
    mode: str

    def __init__(self, pos: NDArray, mode: str):
        super().__init__()
        self.pos = pos
        self.mode = mode

    def apply(self, camera: BaseCamera):
        camera.set_pos(self.pos, self.mode)


class CameraEulerSetpoint(CameraSetpoint):
    pos: Union[NDArray, None]
    euler_angles: NDArray
    mode: str

    def __init__(self, pos: Union[NDArray, None], euler_angles: NDArray, mode='absolute'):
        super().__init__()
        self.pos = pos
        self.euler_angles = euler_angles
        self.mode = mode
        assert(mode == 'absolute' or mode == 'relative')

    def apply(self, camera: BaseCamera):
        if self.pos is not None:
            camera.pos = self.pos

        camera.set_euler_angles(self.euler_angles, mode=self.mode)
