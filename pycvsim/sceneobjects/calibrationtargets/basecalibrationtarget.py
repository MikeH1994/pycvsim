from pycvsim.sceneobjects.sceneobject import SceneObject
import open3d as o3d
from numpy.typing import NDArray
from .utils import transform_object_points


class BaseCalibrationTarget(SceneObject):
    def __init__(self, mesh: o3d.geometry.TriangleMesh, object_points: NDArray):
        super().__init__(mesh)
        self.object_points = object_points

    def object_points(self):
        pos = self.get_pos()
        euler_angles = self.get_euler_angles()
        return transform_object_points(self.object_points, pos, euler_angles)