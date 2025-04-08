
from pycvsim.sceneobjects.sceneobject import SceneObject
import open3d as o3d
from numpy.typing import NDArray
import numpy as np
from pycvsim.targets.utils import transform_object_points
from typing import Tuple
from pycvsim.targets.utils import create_box


class BarTarget(SceneObject):
    boundary_region: NDArray
    object_points: NDArray

    def __init__(self, mesh: o3d.geometry.TriangleMesh, boundary_region: NDArray,
                 name=""):
        super().__init__(mesh, name)
        self.boundary_region = boundary_region

    def get_center(self):
        boundary_region = self.get_boundary_region()
        return np.mean(boundary_region, axis=0)

    def get_boundary_region(self):
        pos = self.get_pos()
        euler_angles = self.get_euler_angles()
        return transform_object_points(self.boundary_region, pos, euler_angles)

    @staticmethod
    def create_target(bar_width=0.4, bar_height=1.0, n_bars=3, target_thickness=0.01,**kwargs) -> Tuple[o3d.t.geometry.TriangleMesh, NDArray]:
        color_1 = (255, 255, 255)
        color_2 = (0, 0, 0)

        for param in kwargs:
            if param == "color_1":
                color_1 = kwargs[param]
            elif param == "color_2":
                color_2 = kwargs[param]
            else:
                raise Exception("Unkown parameter {}".format(param))
        color_1 = np.array(color_1)/255.0
        color_2 = np.array(color_2)/255.0

        mesh = o3d.geometry.TriangleMesh()

        for i in range(n_bars):
            color = color_1 if i % 2 == 0 else color_2
            mesh += create_box(np.array([i*bar_width]), bar_width, bar_height, target_thickness, color)

        return mesh
