from pycvsim.sceneobjects.sceneobject import SceneObject
import open3d as o3d
from numpy.typing import NDArray
from .utils import transform_object_points


class CalibrationTarget(SceneObject):
    boundary_region: NDArray
    object_points: NDArray

    def __init__(self, mesh: o3d.geometry.TriangleMesh, object_points: NDArray, boundary_region: NDArray,
                 name=""):
        super().__init__(mesh, name)
        self.object_points = object_points
        self.boundary_region = boundary_region

    def get_object_points(self):
        pos = self.get_pos()
        euler_angles = self.get_euler_angles()
        return transform_object_points(self.object_points, pos, euler_angles)

    def get_boundary_region(self):
        pos = self.get_pos()
        euler_angles = self.get_euler_angles()
        return transform_object_points(self.boundary_region, pos, euler_angles)
