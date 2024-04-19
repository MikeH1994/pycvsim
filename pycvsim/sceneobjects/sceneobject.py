from __future__ import annotations
import copy
import numpy as np
import open3d as o3d
import panda3d
from numpy.typing import NDArray
from panda3d.core import WindowProperties, NodePath, AntialiasAttrib
import pycvsim.sceneobjects.utils as pycvsim_utils
import pycvsim.core
from typing import List
from scipy.spatial.transform import Rotation


class SceneObject:
    node_path: panda3d.core.NodePath = None
    original_mesh: o3d.geometry.TriangleMesh = None
    n_objects: int = 0
    name: str = ""

    def __init__(self, mesh: o3d.geometry.TriangleMesh, name: str = ""):
        if name == 0:
            name = "Object{}".format(SceneObject.n_objects + 1)
        self.name = name
        geom_node = pycvsim_utils.o3d_mesh_to_pandas3d(mesh)
        node_path = NodePath()
        self.node_path = node_path.attachNewNode(geom_node)
        self.node_path.setTwoSided(True)
        self.node_path.set_antialias(AntialiasAttrib.MAuto)
        self.original_mesh = mesh
        SceneObject.n_objects += 1

    def set_pos(self, pos: NDArray, mode="absolute"):
        assert(mode == "absolute" or mode == "relative")
        if mode == "relative":
            pos += self.get_pos()
        self.node_path.setPos(*pos)

    def get_pos(self):
        return self.node_path.get_pos()

    def set_euler_angles(self, angles: NDArray, mode="absolute"):
        assert(mode == "absolute" or mode == "relative")
        if mode == "relative":
            angles += self.get_euler_angles()
        alpha, beta, gamma = pycvsim.core.xyz_angles_to_panda3d(angles)
        self.node_path.set_hpr(alpha, beta, gamma)

    def get_euler_angles(self):
        # get angles in yxz format
        angles = self.node_path.get_hpr()
        return pycvsim.core.panda3d_angles_to_xyz(angles)

    def mesh(self) -> o3d.geometry.TriangleMesh:
        mesh = copy.deepcopy(self.original_mesh)
        r = mesh.get_rotation_matrix_from_xyz(np.radians(self.get_euler_angles()))
        mesh = mesh.rotate(r, center=(0, 0, 0))
        mesh = mesh.translate(self.get_pos())
        return mesh

    @staticmethod
    def transform(p0: NDArray, p1: NDArray, axes_0: List[NDArray, NDArray, NDArray],
                  axes_1: List[NDArray, NDArray, NDArray]):
        translation = p1 - p0
        vx_1, vy_1, vz_1 = axes_0
        vx_2, vy_2, vz_2 = axes_1

    @staticmethod
    def load_from_file(filepath: str) -> SceneObject:
        mesh = o3d.io.read_triangle_mesh(filepath)
        return SceneObject(mesh)

    @staticmethod
    def load_armadillo():
        armadillo = pycvsim_utils.load_armadillo()
        return SceneObject(armadillo)
