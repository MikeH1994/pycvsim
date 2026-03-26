from pycvsim.sceneobjects.sceneobject import SceneObject
import open3d as o3d
from numpy.typing import NDArray
import numpy as np
from pycvsim.targets.utils import transform_object_points


class CalibrationTarget(SceneObject):
    boundary_region: NDArray
    object_points: NDArray

    def __init__(self, mesh: o3d.geometry.TriangleMesh, object_points: NDArray):
        super().__init__(mesh, "")
        self.object_points = object_points

    def get_object_points(self, transformed=True):
        if transformed:
            return self.object_points @ self.rotation.T + self.pos
        else:
            return np.copy(self.object_points)

    def get_center(self):
        boundary_region = self.get_boundary_region()
        return np.mean(boundary_region, axis=0)

    def get_boundary_region(self):
        pos = self.get_pos()
        euler_angles = self.get_euler_angles()
        return transform_object_points(self.boundary_region, pos, euler_angles)
