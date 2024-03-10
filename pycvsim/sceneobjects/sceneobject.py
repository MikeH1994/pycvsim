from __future__ import annotations
import copy
import open3d as o3d
import panda3d
from panda3d.core import WindowProperties, NodePath, LVecBase3f, AntialiasAttrib
from numpy.typing import NDArray
import numpy as np
import pycvsim.sceneobjects.utils as pycvsim_utils


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

    def set_pos(self, pos: NDArray):
        self.node_path.setPos(*pos)

    def set_euler_angles(self, angles: NDArray):
        alpha, beta, gamma = angles
        self.node_path.set_hpr(beta, gamma, alpha)

    def get_pos(self):
        return self.node_path.get_pos()

    def get_euler_angles(self):
        angles = self.node_path.get_hpr()
        np.array([angles[1], angles[2], angles[0]])
        return angles

    def mesh(self) -> o3d.geometry.TriangleMesh:
        mesh = copy.deepcopy(self.original_mesh)
        r = mesh.get_rotation_matrix_from_xyz(np.radians(self.get_euler_angles()))
        mesh = mesh.rotate(r, center=(0, 0, 0))
        mesh = mesh.translate(self.get_pos())
        return mesh

    @staticmethod
    def load_from_file(filepath: str) -> SceneObject:
        mesh = o3d.io.read_triangle_mesh(filepath)
        return SceneObject(mesh)

    @staticmethod
    def load_armadillo():
        armadillo = pycvsim_utils.load_armadillo()
        return SceneObject(armadillo)
