from __future__ import annotations
import copy
import numpy as np
import open3d as o3d
from numpy.typing import NDArray

import pycv
import pycvsim.sceneobjects.utils as pycvsim_utils
import pycvsim.core
from typing import List
from scipy.spatial.transform import Rotation


class SceneObject:
    mesh: o3d.geometry.TriangleMesh = None
    n_objects: int = 0
    name: str = ""

    def __init__(self, mesh: o3d.geometry.TriangleMesh, name: str = ""):
        if name == 0:
            name = "Object{}".format(SceneObject.n_objects + 1)
        self.name = name
        self.mesh = mesh
        self.pos = np.zeros(3)
        self.rotation = np.eye(3)
        SceneObject.n_objects += 1

    def get_mesh(self) -> o3d.geometry.TriangleMesh:
        mesh = copy.deepcopy(self.mesh)
        m = pycv.create_intrinsic_matrix(self.pos, self.rotation)
        mesh = mesh.transform(m)
        return mesh

    def set_pos(self, pos):
        self.pos = pos

    def set_rotation(self, rotation):
        self.rotation = rotation

    @staticmethod
    def load_from_file(filepath: str) -> SceneObject:
        mesh = o3d.io.read_triangle_mesh(filepath)
        return SceneObject(mesh)

    @staticmethod
    def load_armadillo():
        armadillo = pycvsim_utils.load_armadillo()
        return SceneObject(armadillo)
